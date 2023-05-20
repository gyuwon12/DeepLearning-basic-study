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
    
# AdditiveAttention
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
    
# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.
    In our implementation, we choose the scaled dot-product attention for each head of the multi-head attention.
    nn.MultiheadAttention()은 self-attention도 수행함. 이 class와 다른 조건은 query, key, value tensor shape이 반드시 같아야 한다는 것."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias) 
        
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X : (batch_size, no. of queries or key-value pairs, num_hiddens). 
        # Shape of output X : (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        # transpose_qkv의 결과를 반대로 하기 위한 reverse 함수
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens, window_mask=None):
        # Shape of queries, keys, or values : (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            
        # Shape of output : (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, window_mask)
        # Shape of output_concat : (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
       
# Positional Encoding
class PositionalEncoding(nn.Module):
    """Positional Encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# Position-Wise FFN
class PositionWiseFFN(nn.Module):
    """Some Information about PositionWiseFFN"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        # input shape X : (batch_size, no. of time steps or sequence length in tokens, no. of hidden units or feature dim)
        return self.dense2(self.relu(self.dense1(X)))
        # output shape : (batch_size, no. of time steps, ffn_num_outputs)
        
# Add & Norm
class AddNorm(nn.Module):
    """We can implement the AddNorm class using a residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
        
    def forward(self, X, Y):
        # The residual connection requires that the two inputs are of the same shape 
        # so that the output tensor also has the same shape after the addition operation.
        return self.ln(self.dropout(Y) + X)