import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import argparse
from datasets import cifar10_dataset, cifar100_dataset
import torch
from models import classifier
from utils import utils
from trainers import supervised_joint_trainer
from trainers import tester
from optimizers import optimizers
from torch.utils.data import RandomSampler, DataLoader, Subset
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# parsing arguments
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--epochs", default=200, type=int)
args=parser.parse_args()
batch_size = args.batch
num_epochs = args.epochs

device = utils.get_gpu_if_available()
utils.set_seed(1719)
## build dataset
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()])
cifar10_train_dataset = cifar10_dataset.Cifar10Dataset("train", train_transform, subset_proportion=0.1)
cifar10_test_dataset = cifar10_dataset.Cifar10Dataset("test")
cifar100_train_dataset = cifar100_dataset.Cifar100Dataset("train", train_transform, subset_proportion=0.1)
cifar100_test_dataset = cifar100_dataset.Cifar100Dataset("test")

## dataloaders
cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train_dataset, \
    batch_size=batch_size, shuffle=True, num_workers=2)
cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test_dataset, \
    batch_size=batch_size, shuffle=True, num_workers=2)
cifar100_train_dataloader = torch.utils.data.DataLoader(cifar100_train_dataset, \
    batch_size=batch_size, shuffle=True, num_workers=2)
cifar100_test_dataloader = torch.utils.data.DataLoader(cifar100_test_dataset, \
    batch_size=batch_size, shuffle=True, num_workers=2)

# model
model = classifier.Classifier("resnet18", 10)
model.add_classifier_head(100)

#training scenario1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
train_dataloaders = [cifar10_train_dataloader, cifar100_train_dataloader]
valid_dataloaders = [cifar10_test_dataloader, cifar100_test_dataloader]
task_names = ['Cifar10', 'Cifar100']
trainer = supervised_joint_trainer.SupervisedJointTrainer(train_dataloaders, valid_dataloaders,\
    model, optimizer, device, head_nums=[0, 1], scheduler=scheduler, task_names=task_names)
trainer.run(num_epoch=num_epochs)

print("Acc of cifar10 : ", tester.test(cifar10_test_dataloader, model, 0, device))
print("*"*10)
print("Acc of cifar100 : ", tester.test(cifar100_test_dataloader, model, 1, device))


