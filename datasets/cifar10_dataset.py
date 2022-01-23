import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np

class Cifar10Dataset(Dataset):
    def __init__(self, split="Train", external_transform = None, subset_proportion:float=1):
        if split == "train":
            train_split = True
        elif split == "test":
            train_split= False
        else:
            raise ValueError("Split?")
        
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train_split, download=True)
        internal_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if external_transform is not None:
            self.transform = transforms.Compose([external_transform, internal_transform])
        else:
            self.transform = internal_transform
        
        if subset_proportion < 1:
            indices = np.arange(len(self.dataset))
            train_indices, _ = train_test_split(indices, train_size=subset_proportion, stratify=self.dataset.targets)
            self.dataset.data = self.dataset.data[train_indices]
            self.dataset.targets = np.array(self.dataset.targets)[train_indices]
            

        

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx):
        return self.transform(self.dataset.data[idx]), self.dataset.targets[idx]
