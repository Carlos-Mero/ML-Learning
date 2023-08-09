import torch
from torch import nn
from torch.nn import functional as F

def save_tensors():
    x = torch.arange(4)
    torch.save(x, 'x-file')

def load_tensors():
    x = torch.load('x-file')
    print(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

def save_state_dict():
    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    torch.save(net.state_dict(), 'mlp.params')

def load_model_state_dict():
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    print(clone)

if __name__ == "__main__":
    load_model_state_dict()
