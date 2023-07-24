from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# 모델 로드
model = nn.Linear(3, 1)
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = torch.FloatTensor(request.json['data']) # JSON에서 입력 데이터 추출

    # 모델에 입력 데이터 전달하여 예측 수행
    with torch.no_grad():
        output = model(data)

    # 예측 결과 반환 (예를 들어 JSON 형식으로 반환)
    return jsonify({'prediction': output.tolist()})


if __name__ == '__main__':
    app.run()





