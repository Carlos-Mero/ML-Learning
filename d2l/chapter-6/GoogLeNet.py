import torch
import torch.nn.functional as F
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

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)
    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)

class GoogLeNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(32, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def b3(self):
        return nn.Sequential(
            Inception(32, (48, 64), (8, 16), 16),
            Inception(64, (64, 96), (16, 48), 32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def b4(self):
        return nn.Sequential(
            Inception(96, (48, 104), (8, 24), 32),
            Inception(80, (56, 112), (12, 32), 32),
            Inception(64, (64, 128), (12, 32), 32),
            Inception(128, (80, 160), (16, 64), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def b5(self):
        return nn.Sequential(
            Inception(128, (80, 160), (16, 64), 64),
            Inception(192, (96, 192), (24, 64), 64),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self.b1(),
            self.b2(),
            self.b3(),
            self.b4(),
            self.b5(),
            nn.LazyLinear(10)
        )
    def forward(self, x):
        return self.net(x)

model = GoogLeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

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
    epoches = 20
    print("Now we're training AlexNet on FashionMNIST dataset.")
    print("The model we're using here is:")
    print(model)
    for t in range(epoches):
        print(f"Epoch: {t}\n----------------------------------------------")
        train()
        test()
    print("Done!")
