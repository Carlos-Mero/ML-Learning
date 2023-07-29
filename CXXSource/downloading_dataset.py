import torchvision

# The FashionMNIST dataset
torchvision.datasets.FashionMNIST(
    root="./data/",
    train=False,
    download=True
)

torchvision.datasets.FashionMNIST(
    root="./data/",
    train=True,
    download=True
)
