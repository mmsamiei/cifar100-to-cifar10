import torch

class Optimizer():
    def __init__(self, params, lr=0.00001) -> None:
        self.optimizer = torch.optim.Adam(params, lr=lr)
    
    def step(self):
        return self.optimizer.step()
