import torch

x = torch.arange(12)

print(x)
print(x.shape)
print(x.numel())

r = torch.randn(3,4)
# 这里直接使用了torch.randn()函数，它会返回一个形状为(3,4)的张量，每个元素都是从标准正态分布中随机抽样得到的 

print(r)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(x+y, x-y, x*y, x/y, x**y, sep = ',')

print(torch.exp(x))

X = torch.arange(12, dtype = torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 3, 2, 3], [4, 3, 4, 3]], dtype = torch.float32)

print(torch.cat((X, Y), dim = 0), torch.cat((X, Y), dim = 1), sep = ',')

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))

print(a,b,a+b, sep = '\n')

k = torch.randn(3,4)
l = torch.randn(4,3)

print(torch.mm(k,l))
# 这里调用了torch.mm()函数，它实现了标准的矩阵乘法
# 如果要实现矩阵/向量乘法，可以使用torch.mv()函数

# 后面我们也可以很方便地计算张量的范数，只需要简单地调用torch.norm()函数即可
print(torch.norm(k))
