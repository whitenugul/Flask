from flask import Flask, jsonify, request
import torch
import torch.nn as nn

app = Flask(__name__)

model = nn.Linear(3, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    x_test = torch.FloatTensor(data)
    y_pred = model(x_test)
    return jsonify(y_pred.tolist())

if __name__ == '__main__':
    app.run(debug=True)