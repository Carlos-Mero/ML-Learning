import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = "mps"

batch_size = 128

class HousePrice(Dataset):
    def preprocess(self):
        label = 'SalePrice'
        if label in self.data:
            self.target = self.data.iloc[:, -1].values.astype(np.float32)
            self.data.drop(columns=['Id', 'SalePrice'])
        else:
            self.data.drop(columns=['Id'])
        numeric_features= self.data.dtypes[
            self.data.dtypes !='object'].index
        self.data[numeric_features] = self.data[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        self.data[numeric_features] = self.data[numeric_features].fillna(0)
        self.data = pd.get_dummies(self.data, dummy_na=True)
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        print(self.features)
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.preprocess()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        if self.target is not None:
            target = torch.tensor(self.target[idx])
            return feature, target
        else:
            return feature

train_dataset = HousePrice("./data/train.csv")
test_dataset = HousePrice("./data/test.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(331, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.zeros_(module.weight)
        nn.init.constant_(module.bias, 20000)

model = Model().to(device)
model.linear_relu_stack.apply(init_constant)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X).squeeze()
        loss = loss_fn(torch.log(pred), torch.log(y))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss}, [{current}/{size}]")

def test(dataloader, model, loss_fn):
    pass

if __name__ == "__main__":
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------------------")
        train(train_loader, model, loss_fn, optimizer)
    print("Done!")
