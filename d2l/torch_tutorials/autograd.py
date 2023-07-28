import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

lr = 0.2

for i in range (0, 10):
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Epoch: {i}\n----------------------------------------")
    print(f"loss: {loss}")
    loss.backward()
    g1 = w.grad
    g2 = b.grad
    with torch.no_grad():
        w -= g1 * lr
        b -= g2 * lr
    w.grad.zero_()
    b.grad.zero_()

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print("Finally\n-----------------------------------------")
print(f"final loss: {loss}")
