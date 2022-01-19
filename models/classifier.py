import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet

class Classifier(nn.Module):
    def __init__(self, backbone:str, num_of_class) -> None:
        """
        Args:
            backbone: [resnet18, resnet34, resnet50, resnet101]
            num_of_class: how many classes

        Returns:
            This is a description of what is returned.
        """
        super(Classifier, self).__init__()
        self.num_of_class = num_of_class
        resnet_constructor, self.hid_size = resnet.model_dict[backbone]
        self.backbone = resnet_constructor()
        self.linear = nn.Linear(self.hid_size, self.num_of_class)
    
    def forward(self, x):
        """
        Args:
            x: tensor in shape [batch, 3, 32 ,32] 
        """
        temp = x
        temp = self.backbone(temp)
        temp = self.linear(temp)
        return temp