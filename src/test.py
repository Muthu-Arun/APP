import torch
import torch.nn as nn
import torchvision

model = nn.Sequential(nn.Linear(2, 6, bias=True), nn.ReLU(), nn.Linear(6, 1))
lossfn = nn.MSELoss()
data = [[2, 3], [4, 2], [2, 5], [6, 2], [2, 7]]
x = torch.tensor(data, dtype=torch.float32)
target = [[6], [8], [10], [12], [14]]
y = torch.tensor(target, dtype=torch.float32)

opt = torch.optim.SGD(model.parameters(), lr=0.01)

for epoach in range(30000):
    opt.zero_grad()
    prediction = model(x)

    loss = lossfn(prediction, y)

    loss.backward()
    opt.step()

    print(loss.item())

while True:
    lst = []
    lst.append(float(input()))
    lst.append(float(input()))
    sample = torch.tensor(lst)
    op = model(sample)

    print(op)
