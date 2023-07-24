from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn


class Item(BaseModel):
    data: list

app = FastAPI()

model = nn.Linear(3, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.post('/predict')
def predict(item: Item):
    data = item.data
    x_test = torch.FloatTensor(data)
    y_pred = model(x_test)
    return {"pred": y_pred.tolist()}