import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

batch_size = 128
device = "mps"

train_dataset = datasets.FashionMNIST(
    root="../data",
    train=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="../data",
    train=False,
    transform=ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

def conv_blocks(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_blocks(num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseNet(nn.Module):
    def __init__(self, num_channels=64, growth_rate=32, arch=(4,4,4,4),
                 num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=5, stride=1, padding=2),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        for i, num_convs in enumerate(arch):
            self.net.add_module(
                f'dense_blk{i}', DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(
                    f'tran_blk{i}', transition_block(num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))
    def forward(self, X):
        return self.net(X)

model = DenseNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5,dampening=0.5)

def train():
    size = len(train_dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"Current Training Loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")

def test():
    size = len(test_dataset)
    num_batches = len(test_loader)
    model.eval()
    tloss, tcorrect = 0.0, 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            tloss += loss_fn(pred, y)
            tcorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    tloss /= num_batches
    tcorrect /= size
    tcorrect *= 100
    print(f"Current Test Error: {tloss:>8f}")
    print(f"Current Test Accuracy: {tcorrect:>0.01f}%")

if __name__ == "__main__":
    epoches = 48
    print("Now we're training DenseNet on FashionMNIST dataset.")
    print("The model we're using here is:")
    print(model)
    for t in range(epoches):
        print(f"Epoch: {t}\n----------------------------------------------")
        train()
        test()
    print("Done!")
    torch.save(model, "./DenseNet")
    print("Successfully saved the model in the current directory!")
