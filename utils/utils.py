import torch

def get_gpu_if_available():
    if torch.cuda.is_available():
        print("Hooray! GPU is available!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
