import torch
import random
import numpy as np

def get_gpu_if_available():
    if torch.cuda.is_available():
        print("Hooray! GPU is available!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed_num):
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)


