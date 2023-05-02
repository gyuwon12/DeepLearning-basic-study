"""
def generate_text(model, start_string, sequence_length, device):
    with torch.no_grad():
        # 시작 문자열로부터 첫 입력 시퀀스를 생성
        input_sequence = [char_to_index[ch] for ch in start_string]
        input_sequence = torch.tensor(input_sequence).unsqueeze(0).to(device)

        # 시작 문자열을 출력합니다
        generated_text = start_string

        # 시퀀스 길이만큼 반복합니다
        for i in range(sequence_length):
            # 이전 입력 시퀀스로부터 다음 글자를 예측합니다
            output = model(input_sequence)
            output = torch.softmax(output, dim=2)
            output = output[:, -1, :]
            output_char_index = torch.multinomial(output, num_samples=1).item()
            
            # 다음 입력 시퀀스를 생성합니다
            input_sequence = torch.cat([input_sequence[:, 1:], torch.tensor([[output_char_index]])], dim=1)
            
            # 예측한 글자를 생성된 텍스트에 추가합니다
            generated_text += index_to_char[output_char_index]

        return generated_text
"""