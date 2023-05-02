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
sequence_length = 32
batch_size = 1024
hidden_size = 32 
learning_rate = 0.05
epochs = 100
max_norm = 1.0 # Gradient Clipping을 위한 max_norm 값 설정

def main():
    # GPU 정의
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(777)  
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    # Data Preprocessing
    trainloader = data_preprocessing.make_dataloader(batch_size)
    
    # 모델, 손실 함수, 옵티마이저 정의하기
    #model = models.BiGRU(num_inputs=dedup_word_number, num_hiddens=hidden_size, num_layers=1, num_outputs=dedup_word_number)
    #criterion = loss.My_Loss(device = device)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #clipper = nn.utils.clip_grad_norm_(model.parameters(), max_norm) # Gradient Clipping을 위한 clipper 생성
    
    # Train
    #train.train_model(model, device, dedup_word_number, dataloader, optimizer, clipper, criterion, epochs)
    
    # prediction
    #test.generate_text(model, 'it has', 20, device)
    
main()