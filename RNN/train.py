import torch

def train_model(model, device, num_chars, train_loader, optimizer, clipper, criterion, num_epochs):
    for epoch in range(num_epochs): # 100
        for i, (X, Y) in enumerate(train_loader): # X is input, Y is target 
            X = X.to(device) 
            Y = Y.to(device) # Y shape : (batch, sequence_length)
            outputs = model(X) # outputs shape : (batch, seqeunce_length, 출력(class))의 크기를 가진 3차원 텐서
            loss = criterion(outputs.view(-1, num_chars), Y.view(-1)) # 오차 계산은 펼쳐서 하게 된다 (2D, 1D)로 들어가게 된다.
            # ex) torch.Size([32768, 27]) torch.Size([32768]), 32768 = 1024 * 32
            optimizer.zero_grad() 
            loss.backward()
            clipper 
            optimizer.step()
            
            # 로그 출력
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                    # 진행되는 epoch 표시, 배치 내부 진행 표시, loss 표시
    
    print('End Learning')   