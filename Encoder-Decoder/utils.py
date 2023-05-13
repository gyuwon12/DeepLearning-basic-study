import math
import torch
from torch import nn

# Masking Softmax
def masked_softmax(X, valid_lens):
    """Perform softmax operation by Masking elements on the last axis"""
    # X : 3D tensor (batch_size, sequence_length, feature_dim) feature_dim의 경우 embedding size 이런게 예시
    # valid_lens : 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1) # 2번째 diem
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None] 
        # maxlen 크기의 시퀀스 마스크 생성, 이 마스크는 valid_len의 값을 기준으로 각 위치에 대해 True 또는 False로 채워짐.
        
        X[~mask] = value # ~mask는 마스크의 반대잖아 즉, False들을 택해서 value 값으로 채우겠다는 거임 -> 마스킹 마무리
        return X

    # valid_lens가 None인 경우 - 즉, 마스크를 적용할 필요가 없는 경우 소프트맥스를 수행하고 결과를 반환
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    # None이 아닌 경우 마스킹을 해야함
    else:
        # 만약 valid_lens의 dim이 1이라면, 각 값이 반복되어 shape[1] 크기와 동일한 길이로 만듦
        # 이렇게 하면 각 시퀀스에 대한 유효 길이가 모두 동일하게 됨
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # 그렇지 않은 경우 - 즉, valid_lens의 dim이 2인 경우에는 reshape(-1)을 사용하여 1차원으로 펼쳐준다
        else:
            valid_lens = valid_lens.reshape(-1)
        # 마스킹 적용
        # On the last axis, replace masked elements with a very large negative value, whose exp outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
# Scaled Dot-Product Attention
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, num_heads=None):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads # To be coverd in transformer
        
    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries) => 1D or 2D
    def forward(self, queries, keys, values, valid_lens=None, window_mask=None):
        d = queries.shape[-1] # feature_dim 
        # Swap the last two dimensions of kets with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d) # attention function 부분
        if window_mask is not None: # To be coverd later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # shape of window_mask : (num_windows, no. of queries, no. of key-value pairs)
            scores = scores.reshape((n//(num_windows * self.num_heads),num_windows, self.num_heads, num_queries, num_kv_pairs)) \
            + window_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.reshape((n,num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
class AdditiveAttention(nn.Module):
    """Additive Attention.
    The biggest feature is that the scalar value is the result of the attention score function.
    The dot product attention is the mainstay of modern Transformer architectures. 
    When queries and keys are vectors of different lengths, we can use the additive attention scoring function instead."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.W_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries : (batch_size, no. of queries, 1, num_hiddens)
        # shape of keys : (batch_size, 1, no. of key-value pairs, num_hiddens). Sum them up with bradcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 차원 1을 없애려고 하는
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size, no. of queries, no. of key-value pairs)
        scores = self.W_v(features).squeeze(-1) # scalar 값으로 나온다는게 특징
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)