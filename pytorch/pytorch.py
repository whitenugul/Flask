import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)

optimizer = optim.SGD(model.parameters(), lr=0.00001)


epochs = 1000
for epoch in range(epochs + 1):
  y_pred = model(x_train)
  loss = nn.MSELoss()(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if epoch % 100 == 0:
    print(f' Epoch: {epoch} / {epoch} Loss: {loss:.6f}')


torch.save(model.state_dict(), 'model.pth')

# 파일을 실행 시키고 싶으면 ctrl + Shift + F10을 하면 된다.