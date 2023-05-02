import torch

# test 데이터 원본을 이용한 버전
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