import torch
import torch.nn as nn
from torch.nn import functional as F

# 1995 LeNet
class LeNet(nn.Module):
    """The LeNet-5 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.sigmoid(),
            nn.LazyLinear(84), nn.sigmoid(),
            nn.LazyLinear(num_classes) #softmax
        )        

    def forward(self, x):
        x = self.net(x)
        return x
    
# 2012 AlexNet
class AlexNet(nn.Module):
    """The AlexNet model. The diffrence between Lenet and Alexnet is,
    frist more deeper, second ReLU activation function use."""
    def __init__(self, lr=0.001, num_classes=10):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x

# 2014 VGG
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1)) # h,w 유지
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg_arch():
    # 총 8개의 conv2d layer 인것
    return ((1,64), (1,128), (2,256), (2,512), (2,512)) 

class VGG(nn.Module):
    """The VGG-11(Visual Geometry Group, Conv2d 8 layer, fc 3 layer) model. The idea of using blokcs first emerged from the VGG.
    The key defference is that the convolutional layers are grouped in nonlinear(ReLU) transformations that
    leave the dimensonality unchaged, followed by a resolution-reduction step"""
    def __init__(self, arch, lr=0.001, num_classes=10):
        super(VGG, self).__init__()
        conv_blks = []
        for (num_convs, out_channels) in arch: # 블럭을 튜플로 돌아가겠금
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, 
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x

# 2013 NiN

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU()
    )

class NiN(nn.Module):
    """The NiN(network in network) model.
    The idea behind NiN is to apply a fully connected layer at each pixel location (for each height and width). The resulting 1x1
    convolution can be thought as a fully connected layer acting independently on each pixel location.
    The second significant difference between NiN and both AlexNet and VGG is that NiN avoids fully connected layers altogether."""
    def __init__(self, lr=0.001, num_channels=10):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
        nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(384, kernel_size=3, strides=1, padding=3),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(p=0.5),
        nin_block(num_channels, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

# 2014 GoogleNet
class Inception(nn.Module):
    """Inception blocks."""
    # c1~c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)
        
    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))        
        return torch.cat((b1,b2,b3,b4), dim = 1)
    
class GoogleNet(nn.Module):
    """The GoogleNet model. GoogleNet uses a stack of a total of 9 inception blocks.
    Branch and Concat use are the most diffrence."""
    def __init__(self, lr=0.001, out_channels=10):
        super(GoogleNet, self).__init__()
        self.net = nn.Sequential(
            # d2l상 b1
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # d2l상 b2
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # d2l상 b3 and inception x 2
            Inception(64, (96, 128), (16, 32), 32), # output channels = 64+128+32+32 = 256
            Inception(128, (128, 192), (32, 96), 64), # output channels = 128+192+96+64 = 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # d2l상 b4 and inception x 5
            Inception(192, (96, 208), (16, 48), 64), # output channels = 192+208+48+64 = 512
            Inception(160, (112, 224), (24, 64), 64), # output channels = 160+224+64+64 = 512
            Inception(128, (128, 256), (24, 64), 64), # output channels = 128+256+64+64 = 512
            Inception(112, (144, 288), (32, 64), 64), # output channels = 112+288+64+64 = 528
            Inception(256, (160, 320), (32, 128), 32), # output channels = 256+320+128+32 = 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # d2l상 b5 and inception x 2
            Inception(256, (160, 320), (32, 128), 128), # output channels = 256+320+128+128 = 832
            Inception(384, (192, 384), (48, 128), 128), # output channels = 384+384+128+128 = 1024
            # output channel의 흐름
            # 64 -> 64 -> 192 -> 256 -> 480 -> 512 -> 512 -> 512 -> 528 -> 832 -> 832 -> 1024 -> 10(classes)
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ResNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, x):

        return x

class DenseNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(DenseNet, self).__init__()

    def forward(self, x):

        return x
    