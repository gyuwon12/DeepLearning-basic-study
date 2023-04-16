import torch
import torchvision
import torchvision.datasets as Dataset
import torchvision.transforms as transforms

def get_data_lenet_version():
    root_link = 'MNIST_data/'
    new_transforms=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               ])
    # torchvision.transforms.Compose 클래스: 파라미터로써 compose를 할 Transform 객체의 리스트인 transforms를 받음
    # torchvision.transforms.Resize 클래스: 입력 이미지를 특정 사이즈로 변환함
    # torchvision.transforms.ToTensor 클래스: [0, 255] 범위의 (H X W X C)의 PIL 이미지나 numpy.ndarray를 [0.0, 1.0] 범위의 (C X H X W) float  
    # tensor로 변경함
    
    mnist_train = Dataset.FashionMNIST(root = root_link,
                            train = True,
                            transform = new_transforms,
                            download = True)

    mnist_test = Dataset.FashionMNIST(root = root_link,
                            train = False,
                            transform = new_transforms,
                            download = True)
    
    return mnist_train, mnist_test

def get_data_Imagenet_version():
    root_link = 'MNIST_data/'
    new_transforms=transforms.Compose([
                               transforms.Resize((96, 96)), # Imagenet 데이터셋 크기와 맞게 하려고
                               # (224, 224) -> Alexnet, VGG, Nin
                               # (96, 96) -> GoogleNet, Resnet
                               transforms.ToTensor(),
                               ])
    # torchvision.transforms.Compose 클래스: 파라미터로써 compose를 할 Transform 객체의 리스트인 transforms를 받음
    # torchvision.transforms.Resize 클래스: 입력 이미지를 특정 사이즈로 변환함
    # torchvision.transforms.ToTensor 클래스: [0, 255] 범위의 (H X W X C)의 PIL 이미지나 numpy.ndarray를 [0.0, 1.0] 범위의 (C X H X W) float  
    # tensor로 변경함
    
    mnist_train = Dataset.FashionMNIST(root = root_link,
                            train = True,
                            transform = new_transforms,
                            download = True)

    mnist_test = Dataset.FashionMNIST(root = root_link,
                            train = False,
                            transform = new_transforms,
                            download = True)
    
    return mnist_train, mnist_test

