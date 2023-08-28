import torch
from torch import nn
import datasets
from datasets import load_dataset, Image

device = torch.device("mps");

train_datasets = load_dataset("beans", split="train")
train_datasets = train_datasets.with_format("torch")

print(train_datasets)
