import torch
import torch.nn as nn

class My_Loss(nn.Module):
    """Classfication Loss."""
    def __init__(self, device):
        super().__init__()
        self.Loss = nn.CrossEntropyLoss().to(device)

    def forward(self, hypothesis, Y): # hypothesis = model, Y = real value
        # loss는 (input, target) 이런식으로 계산하게 되는듯
        loss = self.Loss(hypothesis, Y)
        mask = (Y.reshape(-1) != 0).type(torch.float32) # 0인 이유는 <pad>가 임베딩 될 때 0 이거든
        return (loss * mask).sum() / mask.sum()
    
    """
    먼저 모델이 예측한 hypothesis와 정답인 Y 사이의 손실을 계산한다.
    그 다음 부분인 mask = (Y.reshape(-1) != 0).type(torch.float32)에서는, 
    정답 Y에 패딩을 더해 모델이 예측한 값과 크기를 맞춘 후, 이 값이 패딩이 아닌지 판별하는 마스크를 만든다. 
    이렇게 하면 패딩 값을 무시하고 손실을 계산할 수 있다.
    마지막으로, (loss * mask).sum() / mask.sum() 부분에서는 손실과 마스크를 곱한 후, 마스크의 평균값으로 나누어 평균 손실을 계산한다. 
    이렇게 함으로써 패딩을 제외한 실제 값들에 대해서만 손실을 계산하게 된다.
    """