import torch
import numpy as np
import re

# one-hot encoding version
def make_dataloader_ver1(sequence_length, batch_size):
    # 파일 불러오기
    with open("timemachine.txt", "r") as f:
        text = f.read()
    
    # 소문자 데이터로 만들기
    text = re.sub('[^A-Za-z]+', ' ', text).lower() # 가공
    
    # 중복 제거 및 정렬
    chars = list(set(text))
    chars.sort()

    # 각 문자에 고유한 정수 인덱스 부여하기
    char_to_index = {char: index for index, char in enumerate(chars)}
    index_to_char = {index: char for index, char in enumerate(chars)} 
    
    # 입력 데이터와 타겟 데이터 생성
    input_data = []
    target_data = []
    for i in range(0, len(text) - sequence_length):
        input_str = text[i:i+sequence_length]
        target_str = text[i+1:i+sequence_length+1]
        input_data.append([char_to_index[char] for char in input_str])
        target_data.append([char_to_index[char] for char in target_str])
    
    # One-hot encoding
    input_data = [np.eye(len(chars))[x] for x in input_data] # x 데이터는 원-핫 인코딩
    
    # 텐서 변환 -> dataloader에 넣어줘야 하니까
    input_data = torch.FloatTensor(input_data)
    target_data = torch.LongTensor(target_data)
    
    # 데이터셋 및 데이터로더 생성
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # = train_loader
    # print(len(dataloader)) iteration = 170인거지, 17만개 / 1024 = 170 으로 나오는 것.
    
    return dataloader, len(chars)