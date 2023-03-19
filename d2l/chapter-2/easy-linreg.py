import random
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

device = torch.device('mps')

def synthetic_data(w, b, num_examples):
    """Generate y = X w + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X.to(device), y.reshape((-1, 1)).to(device)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
# 前面随机生成数据集的过程与先前完全一致
# 这里我们通过调用框架中现有的API来构造一个迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0, 0.01)
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

net = net.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w.to('cpu') - w.reshape(true_w.shape).to('cpu'))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
