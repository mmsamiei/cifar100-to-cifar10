import torch

class OptimizerBox():
    def __init__(self, optimizer, scheduler) -> None:
        self.optimizer = optimizer
        if scheduler is not None:
            self.has_scheduler = True
            self.scheduler = scheduler
        else:
            self.has_scheduler = False
    
    def step(self):
        return self.optimizer.step()
    
    def scheduler_step(self):
        if self.has_scheduler:
            return self.scheduler.step()
        else:
            raise Exception('Optimizer has no scheduler')
    
    def zero_grad(self):
        return self.optimizer.zero_grad()
