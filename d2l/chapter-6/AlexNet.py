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

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=1),
            nn.LazyConv2d(30, kernel_size=3, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.LazyConv2d(42, kernel_size=2, padding=1), nn.ReLU(),
            nn.LazyConv2d(42, kernel_size=2, padding=1), nn.ReLU(),
            nn.LazyConv2d(30, kernel_size=2, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1), nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.LazyLinear(128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.LazyLinear(10)
        )
    def forward(self, x):
        return self.net(x)

model = AlexNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

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
    epoches = 100
    print("Now we're training AlexNet on FashionMNIST dataset.")
    print("The model we're using here is:")
    print(model)
    for t in range(epoches):
        print(f"Epoch: {t}\n----------------------------------------------")
        train()
        test()
    print("Done!")
    torch.save(model, "./AlexNet")
    print("Successfully saved the model in the current directory!")
