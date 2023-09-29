import torch
from torch import nn

x = torch.rand(10, 10)
y = torch.randn(10)
print(x)
norm = nn.LazyBatchNorm1d()
print(norm(x))
