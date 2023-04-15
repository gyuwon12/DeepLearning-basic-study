import torch

def train_model(model, device, train_loader, optimizer, loss, epochs=20):
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
            my_loss = loss(hypothesis, Y) #오차값 계산
            my_loss.backward() #역전파
            optimizer.step() #최적화 
            # 각 배치마다 계산된 손실값의 평균 구하기 
            avg_loss += my_loss / total_batch
            
        print('[Epoch: {:>2}] loss = {:>.5}'.format(epoch + 1, avg_loss))
    
    print('End Learning')    
    
#Sigmoid -> ReLU로만 바꿔줬는데도 개선이 잘 되네요
#AvgPool -> MaxPool로 바꿔줘도 약간의 효과 기대되긴함
#에폭을 20인데 30-40까지 늘려도 줄어드는 기미가 보일거같기도?
 
# test데이터 원본을 이용한 버전
def test_model_1(model, device, mnist_test):
    accuracy = 0
    #total_batch = len(test_loader)

    with torch.no_grad(): # NOT Training
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
        
def test_model_2(model, device, test_loader):
    accuracy = 0
    total_batch = len(test_loader)

    with torch.no_grad(): # NOT Training + memory save
        model.eval() # model을 evaluation mode로 설정
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            prediction = model(X)
            correct_prediction = torch.argmax(prediction, 1) == Y # Test 진행
            accuracy += correct_prediction.float().mean() # 전체 배치에 대한 정확도 평균을 구하기 위해 해당 과정을 진행
        print('Accuracy:', accuracy.item()/total_batch)
        print('Accuracy = {:>.5}'.format(accuracy.item()/total_batch))
            