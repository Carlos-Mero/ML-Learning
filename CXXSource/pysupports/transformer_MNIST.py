import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

batch_size = 128
device = "mps"

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    transform=ToTensor()
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    transform=ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class tsMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.LazyConv2d(out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=6, kernel_size=1),
            nn.ReLU(), nn.LazyBatchNorm2d(),
            nn.Flatten()
        )
        self.encoder_l = nn.TransformerEncoderLayer(d_model=6 * 28 * 28, nhead=4)
        self.tsencoder = nn.TransformerEncoder(self.encoder_l, 3)
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.tsencoder(x)
        x = self.fc(x)
        return x

model = tsMNIST().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5, dampening=0.5)

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
    epoches = 11
    print("Now we're training DenseNet on FashionMNIST dataset.")
    print("The model we're using here is:")
    print(model)
    for t in range(epoches):
        print(f"Epoch: {t}\n----------------------------------------------")
        train()
        test()
    print("Done!")
    torch.save(model, "../models/MNISTClassification.pt")
    print("Successfully saved the model in the current directory!")
