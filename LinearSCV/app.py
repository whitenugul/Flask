from flask import Flask, request, jsonify
# 백엔드로 통신해야 하기 때문에 request를 import
# 통신은 json으로 하기 때문에 jsonify를 import
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


app = Flask(__name__)

learn_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
learn_label = [0, 0, 0, 1] # 실질적인 정답

# 객체 생성
svc = LinearSVC()

# 학습
svc.fit(learn_data, learn_label)

# Get 방식이 아닌 POST 방식
@app.route('/predict', methods=['POST'])
def predict():
    # post 방식에는 postman 사용
    data = request.json['data']
    pred = svc.predict(data)
    acc = accuracy_score(learn_label, pred)
    return jsonify(acc) # json 형식으로 보낸다.


if __name__ == '__main__':
    app.run(debug=True) # Flask를 띄운다.

