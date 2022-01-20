# from copyreg import constructor
# from mimetypes import init
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: tensor in shape (N, C)
        target: tensor in shape (N)
        """
        return self.criterion(input, target)


