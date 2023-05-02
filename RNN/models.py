import torch
import torch.nn as nn
from torch.nn import functional as F

# 1997 RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.LazyLinear(num_outputs) 

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.fc(output) 
        return output
    
# 1997 LSTM / Not simple version
class LSTMScratch(nn.Module):
    """This model is LSTMScratch. 
    I implemented the calculations of the three gates of lstm and the activation functions at each layer."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super(LSTMScratch, self).__init__()
        self.num_hiddens = num_hiddens
        # Initializing Model Parameters
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens), # R^dxh matrix
                          init_weight(num_hiddens, num_hiddens), # R^hxh matrix
                          nn.Parameter(torch.zeros(num_hiddens))) # 1xh bias
        self.W_xi, self.W_hi, self.b_i = triple() # input gate
        self.W_xf, self.W_hf, self.b_f = triple() # forget gate
        self.W_xo, self.W_ho, self.b_o = triple() # output gate
        self.W_xc, self.W_hc, self.b_c = triple() # Input node
        
    def forward(self, inputs, H_C = None):
        if H_C is None:
            # Inital state with shape: (batch_size, num_hiddens)
            H = torch.zeros((inputs.shape[0], self.num_hiddens), # train 코드상, batch가 first
                            device = inputs.device)
            C = torch.zeros((inputs.shape[0], self.num_hiddens),
                            device = inputs.device)
        else:
            H, C = H_C 
        
        outputs = []
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
            C_tilde = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
            C = (F * C) + (I * C_tilde) # Hadamard(elementwise) product
            H = O * torch.tanh(C) # H_t 
            outputs.append(H) 
            
        return outputs, (H, C) 
    
# 1997 LSTM Concise Implementation 
class LSTM(nn.Module):
    """Some Information about LSTM"""
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(LSTM, self).__init__()
        self.num_hiddens = num_hiddens
        self.lstm = nn.LSTM(num_inputs, num_hiddens, batch_first=True)
        self.fc = nn.LazyLinear(num_outputs)
        
    def forward(self, x):
        # Inital state with shape: (batch_size, num_hiddens) 
        H = torch.zeros((1, x.shape[0], self.num_hiddens), # num_layer=1, batch_first = true 라서, x.shape[0]
                        device = x.device) 
        C = torch.zeros((1, x.shape[0], self.num_hiddens),
                        device = x.device)
        output, (H, C) = self.lstm(x, (H, C))
        output = self.fc(output) 
        return output # output shape = (batch, sequence_length, num_outputs) -> ex (1024, 32, 27)
        # loss에 입력으로 들어갈 때, view metohd로 output은 2D로, y는 1D로 바꿔줘야함 -> train code 내부에서 완료
        
# 2014 GRU / Not simple version
class GRUScratch(nn.Module):
    """Some Information about GRUScratch"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super(GRUScratch, self).__init__()
        self.num_hiddens = num_hiddens
        # Initializing Model Parameters
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens), # R^dxh matrix
                          init_weight(num_hiddens, num_hiddens), # R^hxh matrix
                          nn.Parameter(torch.zeros(num_hiddens))) # 1xh bias
        self.W_xz, self.W_hz, self.b_z = triple() # Update gate
        self.W_xr, self.W_hr, self.b_r = triple() # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple() # Candidate hidden state
        
    def forward(self, inputs, H=None):
        if H is None:
            H = torch.zeros((inputs.shape[0], self.num_hiddens), # batch_first
                            device=inputs.device)
        outputs = []
        for X in inputs:
            Z = torch.sigmoid(torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z)
            R = torch.sigmoid(torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r)
            H_tilde = torch.tanh(torch.matmul(X, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h)
            H = (Z * H) + ((1 - Z) * H_tilde)
            outputs.append(H)
        return outputs, H
    
# 2014 GRU Concise Implementation
class GRU(nn.Module):
    """Some Information about GRU"""
    def __init__(self, num_inputs, num_hiddens, num_layers, num_outputs, dropout=0.0):
        super(GRU, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.gru = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(num_outputs)
        
    def forward(self, x, H=None):
        # Inital state with shape: (batch_size, num_hiddens) -> 굳이 안해줘도 괜찮지만 명시적 전달을 해주려고
        if H is None:
            H = torch.zeros((self.num_layers, x.shape[0], self.num_hiddens), # num_layer=1, batch_first = true 라서, x.shape[0]
                            device = x.device) 
        output, H = self.gru(x, H)
        #output = self.dropout(output)
        output = self.fc(output)
        return output # output shape = (batch, sequence_length, num_outputs) -> ex (1024, 32, 27)
        # loss에 입력으로 들어갈 때, view metohd로 output은 2D로, y는 1D로 바꿔줘야함 -> train code 내부에서 완료

# Bidirection Version with GRU
class BiLSTM(nn.Module):
    """Some Information about BiLSTM"""
    def __init__(self, num_inputs, num_hiddens, num_layers, num_outputs, dropout=0.0):
        super(BiLSTM, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.LazyLinear(num_outputs)
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(num_hiddens*2, num_outputs) # bidirection을 고려한 코드를 이렇게 짜주긴 해야함. lazy method가 편리한 것
        
    def forward(self, x):
        # inputs: (batch_size, sequence_length, num_inputs) -> ex) (1024, 32, 27)
        # output: (batch_size, sequence_length, num_hiddens * 2) -> ex) (1024, 32, 64)
        # hidden: (num_layers * 2, batch_size, num_hiddens) -> ex) (2, 1024, 32)
        output, (H, C) = self.lstm(x)
        # 이 코드는 output과 hidden_concat 사이의 차원을 맞춰주기 위해 unsqueeze(1) 메소드를 사용.
        output = self.fc(output) 
        return output

        # 아래 BiGRU와 비슷한 고려할 점.
        # 이렇게 하면 모든 time_step을 고려한건데, 보통 seq2seq 모델에선 '마지막'time step의 hidden state를 고려한다고함.
        # seq2seq 공부할 때 코드 수정이 약간 필요할지도, 아래와 같이 수정이 가능하긴함. 따로 마지막 step의 forward/backward concat한 결과
        # But, 텐서 자체의 dimention이 변하기 때문에, data preprocessing에서부터 수정이 필요.
        # hidden_concat = torch.cat([H[-2], H[-1]], dim=1) # hidden_concat shape : (batch, num_hiddens * 2) -> ex) (1024, 64)
        
# Bidirection Version with GRU
class BiGRU(nn.Module):
    """Some Information about BiGRU"""
    def __init__(self, num_inputs, num_hiddens, num_layers, num_outputs, dropout=0.0):
        super(BiGRU, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.gru = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(num_outputs)
        
    def forward(self, x, H=None):
        output, H = self.gru(x, H)
        output = self.fc(output)
        return output # output shape = (batch, sequence_length, num_outputs) -> ex (1024, 32, 27)
        # loss에 입력으로 들어갈 때, view metohd로 output은 2D로, y는 1D로 바꿔줘야함 -> train code 내부에서 완료
        