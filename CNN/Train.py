import torch

def train_model(model, device, train_loader, optimizer, criterion, epochs=20):
    total_batch = len(train_loader)
    print('Learning start!')
    for epoch in range(epochs): #20
        avg_loss = 0
        
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad() #기울기 초기화
            # 학습은 [순전파 - 오차값 계산 - 역전파 - 최적화] 의 과정으로 진행!
            hypothesis = model(X) #순전파
            loss = criterion(hypothesis, Y) #오차값 계산
            loss.backward() #역전파
            optimizer.step() #최적화 
            # 각 배치마다 계산된 손실값의 평균 구하기 
            avg_loss += loss.item() / total_batch # loss.item()을 해줘야 메모리 초과 방지 가능
            
        print('[Epoch: {:>2}] loss = {:>.5}'.format(epoch + 1, avg_loss))
    
    print('End Learning')    
    
#Sigmoid -> ReLU로만 바꿔줬는데도 개선이 잘 되네요
#AvgPool -> MaxPool로 바꿔줘도 약간의 효과 기대되긴함
#에폭을 20인데 30-40까지 늘려도 줄어드는 기미가 보일거같기도?
            