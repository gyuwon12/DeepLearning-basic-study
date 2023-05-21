# Library
import torch
import torch.nn as nn
import torch.optim as optim
#from torchsummary import summary

# My module 
import loss
import models
import train
import data_preprocessing
import test

# Hyperparameters
# "Attention is all you need" 내에서의 hyperparameter를 표기.
batch_size = 4096
epochs = 30
sequence_length = 32
num_hiddens = 512
num_blks = 6
ffn_num_hiddens, num_heads = 2048, 8
embed_size = 256
num_layers = 2 
dropout = 0.1
learning_rate = 0.015
max_norm = 1.0 # Gradient Clipping을 위한 max_norm 값 설정

def main():
    # GPU 정의
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(777)  
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(777)
    
    # Data Preprocessing
    trainloader, len_src_vocab, len_tar_vocab = data_preprocessing.make_dataloader(batch_size, sequence_length)
    
    # 모델, 손실 함수, 옵티마이저 정의하기
    seq2seq_encoder = models.Seq2SeqEncoder(len_src_vocab, embed_size, num_hiddens, num_layers, dropout).to(device)
    seq2seq_decoder = models.Seq2SeqDecoder(len_tar_vocab, embed_size, num_hiddens, num_layers, dropout).to(device)
    model = models.Seq2Seq(seq2seq_encoder, seq2seq_decoder).to(device)
    
    transforemr_encoder = models.TransformerEncoder(len_src_vocab, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout).to(device)
    transformer_decoder = models.TransformerDecoder(len_tar_vocab, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout).to(device)
    model = models.Seq2Seq(transforemr_encoder, transformer_decoder).to(device)
    
    criterion = loss.My_Loss(device = device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    clipper = nn.utils.clip_grad_norm_(model.parameters(), max_norm) # Gradient Clipping을 위한 clipper 생성
    
    # Train
    train.train_model(model, device, len_tar_vocab, trainloader, optimizer, clipper, criterion, epochs)

    
main()