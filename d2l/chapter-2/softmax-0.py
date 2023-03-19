import torch
from IPython import display
import FM_load
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = FM_load.load_data_fashion_mnist(batch_size)

num_inputs = 748
num_outputs = 10
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.normal(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # The broadcast mechanism is applied here

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0]), w) + b))

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 模型精度的计算与评估

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval() # Set the model to evaluation mode
    metric = Accumulator(2) # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
