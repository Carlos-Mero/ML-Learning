import torch
from torch import nn

x = torch.linspace(-torch.pi, torch.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3, 4])
xx = x.unsqueeze(-1).pow(p)

model = nn.Sequential(
    nn.Linear(4, 1),
    nn.Flatten(0, 1)
)

loss_fn = nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(f"epoch: {t / 100}")
        print("--------------------------------")
        print(f"current loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

linear_layer = model[0]
print(f"""Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x +
      {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3 +
      {linear_layer.weight[:, 3].item()} x^4""")
