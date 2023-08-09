import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = "mps"

raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train.drop(columns=['Id']).values
train_target = raw_train[:, -1].astype(int)
print(raw_train)
print(train_target)
print(torch.tensor(train_target))
