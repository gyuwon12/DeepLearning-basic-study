import torch
import torch.nn as nn
from torch.nn import functional as F
import collections
import math


import utils

# Encoder - Decoder base architcture
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """We add an additional init_state method to convert the encoder output (enc_all_outputs) into the encoded state."""
    def __init__(self):
        super(Decoder, self).__init__()
        
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
    
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        # Encoder forward 과정
        enc_all_outputs = self.encoder(enc_X, *args) # enc_all_outputs =  encoder_outputs, encoder_state 인 것
        # Encoder -> Decoder 과정
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only, Decoder forward 과정
        return self.decoder(dec_X, dec_state)[0]
        # self.decoder의 output은 tuple 형태야 : (decoder_outputs, decoder_state)
        # train엔 decoder_outputs만 필요하니까 인덱싱을 한것.
    
# Encoder-Decodr Seq2Seq for Machine Translation
def init_seq2seq(module):
    """Initialize weights for seq2seq"""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.wieght)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
                
                
class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        """Seq2Seq Enocder에 해당하는 모델 구조

        Args:
            vocab_size (int): _전체 단어의 사이즈라고 보면된다, 나의 데이터의 경우 인코더에 들어가는 (영어 데이터인 것) vocab_size = 15646._
            embed_size (int): _임베딩 레이어 구조를 정의하기 위한 변수, 128, 256 등의 값들이 사용 됨._
            num_hiddens (int): _RNN layer num_hiddens._
            num_layers (int): _RNn layer num_layers._
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout, bidirectional=False)
        self.apply(init_seq2seq)
        
    def forward(self, X, *args):
        # X shape : (batch_size, sequence_length) -> 예시 : (1024, 32)
        embs = self.embedding(X.t().type(torch.int64)) # why..? transpose? -> decoder에서 context는 "마지막 state"이란 의미가 있음
        # "마지막" sequence를 고려해주기 편하려고 여기서 transfose를 하는것.
        # embs shape : (sequence_length, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape : (sequence_length, batch_size, num_hiddens)
        # state shape : (num_layers, batch_size, num_hiddedns)
        return outputs, state
    
class Seq2SeqDecoder(Decoder):
    """Some Information about Seq2SeqDecoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        """Seq2Seq Dnocder에 해당하는 모델 구조

        Args:
            vocab_size (int): _전체 단어의 사이즈라고 보면된다, 나의 데이터의 경우 인코더에 들어가는 (프랑스어 데이터인 것) vocab_size = 25131_
            embed_size (int): _임베딩 레이어 구조를 정의하기 위한 변수, 128, 256 등의 값들이 사용 됨._
            num_hiddens (int): _RNN layer num_hiddens._
            num_layers (int): _RNn layer num_layers._
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size+num_hiddens이 된 이유 : Decoder의 input이 'context'와 'embedding'을 concat해서 들어가기 때문
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout, bidirectional=False)
        self.fc = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq) # 뭐이게 없긴해도 정상적으로 작동하긴함, 단지 파라미터 초기화 느낌이라
    
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
        
    def forward(self, X, state):
        # X shape : (batch_size, sequence_length) -> 예시 : (1024, 32), X는 decoder input
        # state -> all enocder output : encoder_output, encoder_state 형태
        embs = F.relu(self.embedding(X.t().type(torch.int64)))
        # embs shape : (sequence_length, batch_size, embed_size)
        enc_outputs, hidden_state = state # 이 state는 인코더에서 계산된 문맥 벡터와 마지막 hidden state인 것.
        # context shape : (batch_size, num_hiddens) -> 인코더의 마지막 hidden state와 shape이 같겠지
        context = enc_outputs[-1] # [-1]로 인해 num_layer 부분 dim이 사라져서 (batch_size, num_hiddens)만 남는거지. + 마지막이라는 의미도 있어
        # -> 인코더의 output shape : (sequence_lenth, batch_size, num_hiddens)라서 가능한 인덱싱인 것.
        # Broadcast context to (sequence_lenth, batch_size, num_hiddens) ebms와 context concat을 하기 위해 하는 작업
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature vector
        embs_and_context = torch.cat((embs, context), -1)  # 이게 사실상 최종 Decoder input이라고 볼 수 있어.
        # embs_and_context shape : (sequence_lenth, batch_size, num_hiddens + embed_size)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.fc(outputs).swapaxes(0, 1) # fc layer 통과 후,
        # (sequene_length, batch_size, vocab_size) -> (batch_size, sequence_length, vocab_size)로 만드려고
        # output shape : (batch_size, sequence_length, vocab_size)
        # hidden_state shape : (num_layers, batch_size, num_hiddens)
        return outputs, [enc_outputs, hidden_state]
    
class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, tat_pad=None, lr=None):
        super().__init__(encoder, decoder)
        
# Attention model
class AttentionDecoder(Decoder):
    """The base attention-based decoder interface"""
    def __init__(self):
        super(AttentionDecoder, self).__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
    
class Seq2SeqAttentionDecoder(AttentionDecoder):
    """Some Information about Seq2SeqAttentionDecoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.attention = utils.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout, bidirectional=False)
        self.fc = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, enc_valid_lens):
        # Shape of enc_outputs: (sequence_length, batch_size, num_hiddens)
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state = enc_all_outputs
        return (enc_outputs.permute(1, 0, 2), hidden_state, enc_valid_lens) 
    
    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, sequence_length, num_hiddens)
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state # init_state의 결과물
        # Shape of the X : (sequence_length, batch_size, embed_size), after applying permute method
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query : (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context : (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs : (sequence_length, batch_size, vocab_size)
        outputs = self.fc(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
        # 최종 outputs shape : (batch_size, sequence_length, vocab_size)
    
    @property
    def attention_weights(self):
        return self._attention_weights