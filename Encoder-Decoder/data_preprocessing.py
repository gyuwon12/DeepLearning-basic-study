import torch
from torch.utils.data import DataLoader, TensorDataset
import unicodedata
import re

""" 
Dataset 설명
Tatoeba Project에서 진행한 영어 -> Another Language corpus 데이터셋.
폴더 안에 있는 'fra.txt'는 english -> french version.

Raw data는 다음과 같이 구성되어 있음
< 예시 >
Go.	Va !	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #1158250 (Wittydev)
Go.	Marche.	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #8090732 (Micsmithel)
Go.	En route !	CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #8267435 (felix63)
여기선 차이가 잘 안나지만, 각 공백의 구분이 '\t'로 구성된 형태이다.
load_preprocessed_data() 함수에서 이를 적용해 src_line, tar_line으로 'CC-' 부분을 제외하고 처리한다.

Tensor 형태의 data form을 만드는 것은 build_array() 함수에서 최종적으로 진행된다.
출력해보니 영어 데이터의 vocab_size = 14969, 프랑스어는 24980으로 나왔다. 단어 수가 굉장히 많다.

최종 학습 데이터 구조는 아래와 같다.
encoder input : torch.Size([217975, 8])
decoder input : torch.Size([217975, 16])
decoder target : torch.Size([217975, 16])

make_dataloader() 함수에서 위 3가지 tensor가 minibatch로 수정되어 훈련이 되도록 한다.
"""

def to_ascii(s):
    # 프랑스어 악센트(accent) 삭제
    # 예시 : 'déjà diné' -> deja dine
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    # 악센트 제거 함수 호출
    sent = to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백 추가.
    # ex) "I am a student." => "I am a student ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    # 다수 개의 공백을 하나의 공백으로 치환
    sent = re.sub(r"\s+", " ", sent)
    return sent

def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open('fra.txt', encoding='utf-8') as lines:
        for i, line in enumerate(lines):
            # source 데이터와 target 데이터 분리
            src_line, tar_line, _ = line.strip().split('\t')

            # source 데이터 전처리
            src_line = preprocess_sentence(src_line)
            src_line = [w for w in (src_line + " <eos>").split()]

            # target 데이터 전처리 -> decoder에 들어갈건, decoder input에 <sos>, decoder output에 <eos> 추가
            tar_line = preprocess_sentence(tar_line)
            tar_line_in = [w for w in ("<sos> " + tar_line).split()]
            tar_line_out = [w for w in (tar_line + " <eos>").split()]

            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)

    return encoder_input, decoder_input, decoder_target

def padding(sentences, num_steps):
    # 부족한 sequence length 만큼 word 단위의 빈 공간을 <pad>로 채울 정규표현식
    pad_or_trim = lambda seq, t: (seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
    # 적용
    sentences = [pad_or_trim(s, num_steps) for s in sentences]
    return sentences

def encoded_sentences(sentences):
    # 단어장 생성
    vocab = set()
    for sent in sentences:
        for word in sent:
            vocab.add(word)
    vocab_size = len(vocab)
    valid_vocab_size = len(vocab) - 2 # encode input은 <eos>와 <pad>를 제외한 것 나머지도 마찬가지로 <sos>와 <pad> / <eos>와 <pad>를 뺀 값

    # 단어장 인덱스 매핑
    word_to_index = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for i, word in enumerate(vocab):
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
    
    # 정수 인코딩
    encoded_sentences = []
    for sent in sentences:
        encoded_sent = [word_to_index[word] for word in sent]
        encoded_sentences.append(encoded_sent)
        
    # tensor로 변환
    encoded_sentences = torch.LongTensor(encoded_sentences)
    return encoded_sentences, vocab_size, valid_vocab_size

def build_array():
    encoder_input, decoder_input, decoder_target = load_preprocessed_data()
    padding_en_input = padding(encoder_input, 8)
    padding_de_input = padding(decoder_input, 16)
    padding_de_target = padding(decoder_target, 16)
    
    encoded_en_input, src_vocab, src_valid_len = encoded_sentences(padding_en_input)
    encoded_de_input, _, _ = encoded_sentences(padding_de_input)
    encoded_de_target, _, _ = encoded_sentences(padding_de_target)

    return encoded_en_input, encoded_de_input, encoded_de_target

def make_dataloader(batch_size):
    encoder_input, decoder_input, decoder_target = build_array()
    # 데이터셋이 이미 텐서 형태라서, 데이터셋을 래핑하지 않고 데이터를 불러옵니다.
    train_data = TensorDataset(encoder_input, decoder_input, decoder_target)
    
    # 데이터 로더 생성
    BATCH_SIZE = batch_size # 배치 크기
    trainloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True, # 셔플 사용 여부
        drop_last=True # 마지막 배치를 버릴지 여부 
        ) 
    
    return trainloader

"""
seq2seq 모델의 학습 데이터셋은 보통 인코더의 입력 데이터와 디코더의 입력 데이터, 그리고 디코더의 출력 데이터로 이루어져 있습니다.

인코더의 입력 데이터는 문장을 토큰화하고 정수 인코딩한 행렬로, 일반적으로 shape은 (batch_size, max_encoder_sequence_length) 입니다. 이 때 max_encoder_sequence_length는 인코더의 입력 시퀀스 중 가장 긴 시퀀스의 길이입니다.

디코더의 입력 데이터는 인코더의 출력인 context vector와 문장의 시작을 알리는 start token을 결합한 행렬입니다. 이 역시 shape은 (batch_size, max_decoder_sequence_length)으로, max_decoder_sequence_length는 디코더의 입력 시퀀스 중 가장 긴 시퀀스의 길이입니다.

마지막으로 디코더의 출력 데이터는 디코더의 입력과 마찬가지로 문장을 토큰화하고 정수 인코딩한 행렬로, shape은 (batch_size, max_decoder_sequence_length)입니다.

따라서, seq2seq 모델의 학습 데이터셋 전체 shape은 (batch_size, max_encoder_sequence_length), (batch_size, max_decoder_sequence_length), (batch_size, max_decoder_sequence_length)으로 이루어진 튜플 형태가 됩니다.
"""