# Library
import torch
import torch.nn as nn
import torch.optim as optim
#from torchsummary import summary

# My module    
import Loss
import Model
import Dataloader
import Datadownload
import Train

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
    row_train_data, row_test_data = Datadownload.get_data_Imagenet_version()
    train_loader, test_loader = Dataloader.make_dataloader(row_train_data, row_test_data, batch_size)
    
    
    # Model 정의, 원하는 모델을 불러와서 쓸것.
    model = Model.DenseNet().to(device) # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = Model.VGG(arch=Model.vgg_arch()).to(device) # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model)
    
    # Loss와 Optim 정의
    My_loss = Loss.My_Loss(device = device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_batch = len(train_loader)
    print('총 배치의 수 : {}'.format(total_batch))
    # 60000/128 = 468.x
    
    # Train
    Train.train_model(model, device, train_loader, optimizer, My_loss, epochs)
    
    # Test
    Train.test_model_2(model, device, test_loader)
    
main()