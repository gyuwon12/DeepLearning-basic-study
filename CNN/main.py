# Library
import torch
import torch.nn as nn
import torch.optim as optim
#from torchsummary import summary

# My module    
import loss
import models
import dataloader
import datadownload
import train
import test

# Hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 128

def main():
    # GPU 정의
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(777)  
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    print(device)
        
    # DownLoading Dataset and Make Dataloader
    row_train_data, row_test_data = datadownload.get_data_Imagenet_version()
    train_loader, test_loader = dataloader.make_dataloader(row_train_data, row_test_data, batch_size)
    
    
    # Model 정의, 원하는 모델을 불러와서 쓸것.
    model = models.DenseNet().to(device) 
    #model = Model_.VGG(arch=Model.vgg_arch()).to(device) 
    #print(model)
    
    # Loss와 Optim 정의
    criterion = loss.My_Loss(device = device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_batch = len(train_loader)
    print('총 배치의 수 : {}'.format(total_batch))
    # 60000/128 = 468.x
    
    # Train
    train.train_model(model, device, train_loader, optimizer, criterion, epochs)
    
    # Test
    test.test_model_2(model, device, test_loader)
    
main()