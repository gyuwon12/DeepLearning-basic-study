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
batch_size = 128
epochs = 30
sequence_length = 32
num_hiddens = 256
embed_size = 256
num_layers = 2 
dropout = 0.2
learning_rate = 0.005

def main():
    # GPU 정의
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(777)  
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(777)
    
    # Data Preprocessing
    trainloader, len_src_vocab, len_tar_vocab = data_preprocessing.make_dataloader(batch_size, sequence_length)
    
    # 모델, 손실 함수, 옵티마이저 정의하기
    encoder = models.Seq2SeqEncoder(len_src_vocab, embed_size, num_hiddens, num_layers, dropout).to(device)
    decoder = models.Seq2SeqDecoder(len_tar_vocab, embed_size, num_hiddens, num_layers, dropout).to(device)
    model = models.Seq2Seq(encoder, decoder).to(device)

    criterion = loss.My_Loss(device = device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train
    train.train_model(model, device, len_tar_vocab, trainloader, optimizer, criterion, epochs)
    
    # prediction
    #test.generate_text(model, 'it has', 20, device)
    
main()