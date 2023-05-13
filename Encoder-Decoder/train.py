import torch

def train_model(model, device, len_tar_vocab, train_loader, optimizer, clipper, criterion, num_epochs):
    for epoch in range(num_epochs): 
        for i, (encoder_inputs, decoder_inputs, decoder_targets)  in enumerate(train_loader): 
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device) # decoder_target shape : (batch_size, sequence_length) 에시 : (1024, 32)
            
            # Forward
            decoder_outputs = model(encoder_inputs, decoder_inputs) # 곧 예측값
            # decoder_outputs shape : (batch_size, sequence_length, vocab_size)
            
            # loss 계산
            loss = criterion(decoder_outputs.reshape(-1, len_tar_vocab), decoder_targets.view(-1)) # loss 계산을 위해 2D, 1D
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