from __future__ import annotations
from typing import Dict, List, Tuple
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

"""The LoadCifar10() method below downloads the Cifar-10 dataset and prep the data for learning and inference."""
def LoadCifar10(DataDirectory: str = "./data"):
    MeanTuple = (0.4914, 0.4822, 0.4465)
    StdTuple = (0.2470, 0.2435, 0.2616)
    TrainTransform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MeanTuple, StdTuple),
    ])
    TestTransform = T.Compose([
        T.ToTensor(),
        T.Normalize(MeanTuple, StdTuple),
    ])
    TrainDataset = torchvision.datasets.CIFAR10(root=DataDirectory, train=True, download=True, transform=TrainTransform)
    TestDataset = torchvision.datasets.CIFAR10(root=DataDirectory, train=False, download=True, transform=TestTransform)
    return TrainDataset, TestDataset

