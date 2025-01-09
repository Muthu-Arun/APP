import torch
import torch.nn as nn

model = nn.Linear(2, 1)
lossfn = nn.MSELoss()
data = [[2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]
x = torch.tensor(data, dtype=torch.float32)
target = [[6], [8], [10], [12], [14]]
y = torch.tensor(target, dtype=torch.float32)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

cycle = 10000
for i in range(cycle):
    optimizer.zero_grad()
    prediction = model(x)

    loss = lossfn(prediction, y)

    loss.backward()
    optimizer.step()

    print(loss.item())

while True:
    lst = []
    lst.append(float(input()))
    lst.append(float(input()))
    sample = torch.tensor(lst)
    op = model(sample)

    print(op)
