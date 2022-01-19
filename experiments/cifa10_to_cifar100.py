from datasets import cifar10_dataset, cifar100_dataset
import torch
from models import classifier
from utils import utils
from trainers import supervised_trainer
from trainers import tester
from optimizers import optimizers

device = utils.get_gpu_if_available()

## build dataset
cifar10_train_dataset = cifar10_dataset.Cifar10Dataset("train")
cifar10_test_dataset = cifar10_dataset.Cifar10Dataset("test")
cifar100_train_dataset = cifar100_dataset.Cifar100Dataset("train")
cifar100_test_dataset = cifar100_dataset.Cifar100Dataset("test")

## dataloaders
cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train_dataset, \
    batch_size=128, shuffle=True, num_workers=2)
cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test_dataset, \
    batch_size=128, shuffle=True, num_workers=2)
cifar100_train_dataloader = torch.utils.data.DataLoader(cifar100_train_dataset, \
    batch_size=128, shuffle=True, num_workers=2)
cifar100_test_dataloader = torch.utils.data.DataLoader(cifar100_test_dataset, \
    batch_size=128, shuffle=True, num_workers=2)

# model
model = classifier.Classifier("resnet18", 10)
model.add_classifier_head(100)


optimizer = optimizers.Optimizer(model.parameters(), lr=0.00001)
trainer = supervised_trainer.SupervisedTrainer(cifar10_train_dataloader, \
    cifar10_test_dataloader, model, optimizer, device, head_num=0)
trainer.run(num_epoch=50)
print("Acc of cifar10 : ", tester.test(cifar10_test_dataloader, model, 0, device))


optimizer = optimizers.Optimizer(model.parameters(), lr=0.00001)
trainer = supervised_trainer.SupervisedTrainer(cifar100_train_dataloader, \
    cifar100_test_dataloader, model, optimizer, device, head_num=1)
trainer.run(num_epoch=50)
print("Acc of cifar100 : ", tester.test(cifar100_test_dataloader, model, 0, device))
