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
    def __init__(self, lr=0.001, out_classes=10):
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
            nn.LazyLinear(out_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x

# 2015 ResNet
class Residual(nn.Module):
    """The residual block has two 3x3 convolutional layers with the same number of output channels. 
    Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. 
    Then, we skip these two convolution operations and add the input directly before the final ReLU activation function. 
    This kind of design requires that the output of the two convolutional layers has to be of the same shape as the input, 
    so that they can be added together."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride = strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv: # channel 개수 조정을 위한 용도
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 3x3 conv2d -> batchnorm -> activation
        out = self.bn2(self.conv2(out)) # 3x3 conv2d -> batcnorm
        if self.conv3:
            x = self.conv3(x) # channel수 조절 용도 + H/W를 줄이는 stride=2 넣어주는 용도로 1x1 conv2d를 이용함
        out += x # skip layer = f(x)(==out) + x
        return out

class ResNet(nn.Module):
    """The ResNet model. The first two layers of ResNet are the same as those of the GoogLeNet we described before.
    GoogLeNet uses four modules made up of Inception blocks. However, ResNet uses four modules made up of residual blocks, 
    each of which uses several residual blocks with the same number of output channels.
    Then, we add all the modules to ResNet. Here, two residual blocks are used for each module. 
    Lastly, just like GoogLeNet, we add a global average pooling layer, followed by the fully connected layer output."""
    # block function
    def block(self, num_residuals, num_channels, first_block=False): 
        # First block 의미는 "블럭 기준" 처음 Residual block에 1x1 conv2d layer을 사용하지 않겠다는 거임 
        blocks=[]
        for i in range(num_residuals):
            if i==0 and not first_block: # i==0 의미는 "블럭 내부 기준" 처음을 의미하는 것
                blocks.append(Residual(num_channels, use_1x1conv=True, strides=2)) 
            else:
                blocks.append(Residual(num_channels))
        return nn.Sequential(*blocks)
    
    # initialization    
    def __init__(self, arch, lr=0.001, num_classes=10):
        super(ResNet, self).__init__()
        # ResNet 시작의 2개 layer은 GoogleNet과 똑같아(batchnorm 빼고)
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Residual block 시작하는 부분
        for i, b in enumerate(arch):
            #i는 Len(arch), *b => (2,64)와 같은 (num_residual, middle, out)이 한번에 감! *의 효과  
            self.net.add_module(f'{i+4}', self.block(*b, first_block=(i==0)))
        # Lastly, just like GoogLeNet, we add a global average pooling layer, followed by the fully connected layer output.
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes) # 10
        ))
        
    def forward(self, x):
        x = self.net(x)
        return x

class ResNet18(ResNet):
    """The ResNet18 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes)
        
class ResNet34(ResNet):
    """The ResNet18 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__(((3, 64), (4, 128), (6, 256), (3, 512)), lr, num_classes)

#=========================================================================================================
class Residual_50up(nn.Module):
    def __init__(self, middle_channels, out_channels, use_1x1conv=False, strides=1):
        super(Residual_50up, self).__init__()
        self.conv1 = nn.LazyConv2d(middle_channels, kernel_size=1, padding=0, stride=strides) 
        self.conv2 = nn.LazyConv2d(middle_channels, kernel_size=3, padding=1)
        self.conv3 = nn.LazyConv2d(out_channels, kernel_size=1, padding=0) # padding=0으로 해줘야 H, W 유지
        if use_1x1conv: # channel 개수 조정을 위한 용도
            self.conv4 = nn.LazyConv2d(out_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.iden = nn.LazyConv2d(out_channels, kernel_size=1, stride=strides)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 1x1 conv2d -> batchnorm -> activation
        out = F.relu(self.bn2(self.conv2(out))) # 3x3 conv2d -> batcnorm
        out = self.bn3(self.conv3(out))
        if self.conv4:
            x = self.conv4(x) # channel수 조절 용도 + H/W를 줄이는 stride=2 넣어주는 용도로 1x1 conv2d를 이용함
        if out.shape != x.shape: # Residual block에서 middle and out channel이 다르기에 이를 보정하기 위한 layer!!
            x = self.iden(x)
        out += x # skip layer = f(x)(==out) + x
        return F.relu(out) 

class ResNet_50up(nn.Module):
    # block function
    def block(self, num_residuals, middle_channels, out_channels, first_block=False): 
        # First block 의미는 "블럭 기준" 처음 Residual block에 1x1 conv2d layer을 사용하지 않겠다는 거임 
        blocks=[]
        for i in range(num_residuals):
            if i==0 and not first_block: # i==0 의미는 "블럭 내부 기준" 처음을 의미하는 것
                blocks.append(Residual_50up(middle_channels, out_channels, use_1x1conv=True, strides=2)) 
            else:
                blocks.append(Residual_50up(middle_channels, out_channels))
        return nn.Sequential(*blocks)
    
    # initialization    
    def __init__(self, arch, lr=0.001, num_classes=10):
        super(ResNet_50up, self).__init__()
        # ResNet 시작의 2개 layer은 GoogleNet과 똑같아(batchnorm 빼고)
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Residual block 시작하는 부분
        for i, b in enumerate(arch): # i는 Len(arch), *b => (3,64,256)와 같은 (num_residual, middle, out)이 한번에 감! *의 효과  
            self.net.add_module(f'{i+4}', self.block(*b, first_block=(i==0)))
        # Lastly, just like GoogLeNet, we add a global average pooling layer, followed by the fully connected layer output.
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes) # 10
        ))
        
    def forward(self, x):
        x = self.net(x)
        return x 
    
class ResNet50(ResNet_50up):
    """The ResNet50 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__(((3, 64, 256), (4, 128, 512), (6, 256, 1024), (3, 512, 2048)), lr, num_classes)

class ResNet101(ResNet_50up):
    """The ResNet101 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__(((3, 64, 256), (4, 128, 512), (23, 256, 1024), (3, 512, 2048)), lr, num_classes)

class ResNet152(ResNet_50up):
    """The ResNet152 model."""
    def __init__(self, lr=0.001, num_classes=10):
        super().__init__(((3, 64, 256), (8, 128, 512), (36, 256, 1024), (3, 512, 2048)), lr, num_classes)

# 2017 ResNeXt
class ResNeXtBlock(nn.Module):
    """The ResNeXt Block. Inception module in GoogleNet + Residual block.
    Different from the smorgasbord of transformations in Inception, 
    ResNeXt adopts the same transformation in all branches, thus minimizing the need for manual tuning of each branch."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False, strides=1):
        super(ResNeXtBlock, self).__init__()
        bot_channels = int(round(num_channels * bot_mul)) # itermediate channels
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3, stride=strides, padding=1, groups=bot_channels/groups)
        self.conv3 = nn.LaztConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.conv4:
            x = self.bn4(self.conv4(x)) # shape 맞추기 용도
        return F.relu(out + x)

# 2018 DenseNet
def dense_conv_block(num_channels):
    """DenseNet uses the modified “batch normalization, activation, and convolution” structure of ResNet."""
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
    )

# DenseNet은 크게 2개의 레이어로 구성댐
# 1.concatenate하는 용도 
# 2.control the number of channels를 위한 목적 (concatenate때문에 채널수가 커지니까)
class DenseBlock(nn.Module):
    """The key difference between ResNet and DenseNet is that in the latter case outputs are concatenated."""
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(dense_conv_block(num_channels))
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        for block in self.net:
            out = block(x)
            # Concatenate input and output of each block along the channels
            out = torch.cat((out, x), dim=1) # 여기가 resnet과의 차이
        return out
 
def transition_block(num_channels):
    """A transition layer is used to control the complexity of the model. It reduces the number of channels by using an 1x1
    convolution. Moreover, it halves the height and width via average pooling with a stride of 2."""
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseNet(nn.Module):
    """The DenseNet model. DenseNet first uses the same single convolutional layer and max-pooling layer as in ResNet.
    Then, similar to the four modules made up of residual blocks that ResNet uses, DenseNet uses four dense blocks.
    Here, we use the transition layer to halve the height and width and halve the number of channels. 
    Similar to ResNet, a global pooling layer and a fully connected layer are connected at the end to produce the output."""
    def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4), lr=0.001, num_classes=10):
        super(DenseNet, self).__init__()
        # 처음 부분은 ResNet과 같음
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        # Block section
        for i, num_convs in enumerate(arch): # 0 4 -> 1 4 -> 2 4 -> 3 4
            self.net.add_module(f'dense_block{i+1}', DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate # Dense block 거치면서 output 증가되는 것 적용
            # A transition layer that halves the number of channels is added between the dense blocks
            if i != len(arch) - 1: 
                num_channels //= 2
                self.net.add_module(f'transition_block{i+1}', transition_block(num_channels))
        # last layer part
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes) # 10
        ))

    def forward(self, x):
        x = self.net(x)
        return x
    