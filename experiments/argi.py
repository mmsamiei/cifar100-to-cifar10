import argparse


batch_size = 256
cifar10_epochs = 100
cifar100_epochs = 100


parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("--batch",default=batch_size, type=int)
parser.add_argument("--first_epochs",default=cifar10_epochs, type=int)
parser.add_argument("--second_epochs",default=cifar100_epochs, type=int)
args=parser.parse_args()
batch_size = args.batch
cifar10_epochs = args.first_epochs
cifar100_epochs = args.second_epochs