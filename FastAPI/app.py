from fastapi import FastAPI
import torch
import torch.nn as nn
from fastapi import Request

app = FastAPI()

@app.get("/")
def index():
    return {"index": "Hello FastAPI"} # 자동적으로 json 형태로 인식이 된다.


@app.get("/math/sum")
def math_sum(num1: int, num2: int):
    return {"result": num1 + num2}



model = nn.Linear(3, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()
@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    data = data["data"]
    x_test = torch.FloatTensor(data)
    y_pred = model(x_test)
    return {
        "pred": y_pred.tolist()
    }
