import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# preprocess
features = pd.concat(
    (train_data.drop(columns=['Id', 'SalePrice']),
    test_data.drop(columns=['Id'])))
num_features = features.dtypes[features.dtypes!='object'].index
features[num_features] = features[num_features].apply(
    lambda x: (x-x.mean()) / (x.std()))
features[num_features].fillna(features[num_features].mean(), inplace=True)
features = pd.get_dummies(features, dummy_na=True)
td = features[:train_data.shape[0]].copy()
td['SalePrice'] = train_data['SalePrice']
val = features[train_data.shape[0]:].copy()
get_tensor = lambda x: torch.tensor(
    x.values.astype(float), dtype=torch.float32)
ttarget = torch.log(get_tensor(td['SalePrice'])).reshape((-1, 1))
td = get_tensor(td.drop(columns=['SalePrice']))
val = get_tensor(val)
fold_size = td.shape[0] // 6
td_t = (td[fold_size:], ttarget[fold_size:].flatten())
td_v = (td[:fold_size], ttarget[:fold_size].flatten())

def conv_blocks(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        nn.LazyConv1d(kernel_size=1, out_channels=num_channels)
    )

class res_block(nn.Module):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        layer = []
        for _ in range(num_convs):
            layer.append(conv_blocks(num_channels))
        self.net = nn.Sequential(*layer)
        self.trans = conv_blocks(num_channels)
    def forward(self, x):
        y = self.net(x)
        x = self.trans(x)
        y += x
        return F.relu(y)

class trans_block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyBatchNorm1d(), nn.ReLU(),
            nn.LazyConv1d(out_channels=num_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=3)
        )
    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv1d(out_channels=16, kernel_size=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            res_block(num_convs=2, num_channels=8),
            trans_block(4),
            res_block(num_convs=2, num_channels=4),
            trans_block(2),
            res_block(num_convs=2, num_channels=2),
            trans_block(1),
            nn.Flatten(),
            nn.LazyLinear(24),
            nn.ReLU(),
            nn.LazyLinear(1),
        )
    def forward(self, x):
        return self.net(x)

# train the model
device = 'mps'
model = Net().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.5)
td_t = ((td_t[0].to(device), td_t[1].to(device)))
td_v = (td_v[0].to(device), td_v[1].to(device))
tr_loss = []
vl_loss = []

# DataLoader
batch_size = 128

class house_price_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return (self.data[0][index], self.data[1][index])
    def __len__(self):
        return len(self.data[1])

tdt_set = house_price_dataset(td_t)
tdv_set = house_price_dataset(td_v)

train_loader = DataLoader(tdt_set, batch_size=batch_size)
test_loader = DataLoader(tdv_set, batch_size=batch_size)

def tr():
    size = len(td_t[0])
    csize = 0
    model.train()
    for (X, y) in train_loader:
        X = X.to(device).unsqueeze(1)
        y = y.to(device)
        pred = model(X).flatten()
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        tr_loss.append(loss)
        csize += len(y)

        print(f"Current Training Loss is: {loss}, [{csize}/{size}]")

def vl():
    model.eval()
    loss = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to(device).unsqueeze(1)
            y = y.to(device)
            pred = model(X).flatten()
            loss += loss_fn(pred, y)
            vl_loss.append(loss)
    print("----------------------------------------------------")
    print(f"Current Test Loss is: {loss}")

if __name__ == "__main__":
    epoches = 48
    for epoch in range(epoches):
        print(f"Epoch: {epoch}")
        print("----------------------------------------------------")
        tr()
        vl()
    print("Done!")
