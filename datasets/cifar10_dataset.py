import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class Cifar10Dataset(Dataset):
    def __init__(self, split="Train"):
        if split == "train":
            train_split = True
        elif split == "test":
            train_split= False
        else:
            raise ValueError("Split?")
        
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train_split, download=True)
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset.data[idx]), self.dataset.targets[idx]
