import torch
from torch import nn

w_t = torch.randn(10, 10)
b_t = torch.randn(10)
x = torch.randn(2000, 10)
print(f"The true w is\n{w_t}")
print(f"The true b is\n{b_t}")
y = torch.matmul(x, w_t) + b_t.unsqueeze(0)
print(f"The corresponding output is\n{y.size()}")

class LinearRegressionScratch(nn.Module):
    """The module is used for linear regression"""
    def __init__(self):
        super().__init__()
        self.linear_reg = nn.Sequential(
            nn.Linear(10, 10)
        )

    def forward(self, X):
        return self.linear_reg(X)

loss_fn = nn.MSELoss(reduction='mean')
learning_rate = 0.03
model = LinearRegressionScratch()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

def train(model: LinearRegressionScratch,
          loss_fn: nn.MSELoss,
          optimizer: torch.optim.SGD):
    model.train()
    epochs = 10000
    for t in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (t % 100 == 0):
            print(f"Epoch {t / 100}")
            print("--------------------------------------")
            print(f"The current loss is: {loss}")

if __name__ == "__main__":
    train(model, loss_fn, optimizer)
    print("Done!")
    print(f"The true w is\n{w_t}")
    print(f"The true b is\n{b_t}")
    print(f"The predicted parameters are")
    for params in model.parameters():
        print(params)
