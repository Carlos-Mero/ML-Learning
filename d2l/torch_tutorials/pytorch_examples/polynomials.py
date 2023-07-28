import torch
import numpy as np

x = torch.linspace(-torch.pi, torch.pi, 2000)
y = torch.sin(x)

params = torch.randn(5, requires_grad=True)
pows = torch.linspace(0, 4, 5)
learning_rate = 3e-4

for i in range(50001):
    pred = torch.zeros(2000)
    for t in range(5):
        pred += params[t] * x ** t

    loss_func = torch.nn.MSELoss()

    ls = loss_func(pred, y)
    ls.backward()

    if i % 100 == 0:
        print(f"Epoch: {i / 100}")
        print("----------------------------")
        print(f"current training loss: {ls}")
        print(f"""polynomial: {params[0]} + {params[1]}x + {params[2]}x^2
              + {params[3]}x^3 + {params[4]}x^4""")

    with torch.no_grad():
        params -=  params.grad * learning_rate
        params.grad.zero_()
