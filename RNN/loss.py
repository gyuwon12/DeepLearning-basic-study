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
        return loss