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
        self.num_of_class = [num_of_class]
        resnet_constructor, self.hid_size = resnet.model_dict[backbone]
        self.backbone = resnet_constructor()
        self.linears = nn.ModuleList([nn.Linear(self.hid_size, self.num_of_class[-1])])
    
    def forward(self, x):
        """
        Args:
            x: tensor in shape [batch, 3, 32 ,32] 
        """
        temp = x
        temp = self.backbone(temp)
        temps = []
        for i in range(len(self.linears)):
            temps.append(self.linears[i](temp))
        return temps

    def add_classifier_head(self, num_of_class):
        self.num_of_class.append(num_of_class)
        self.linears.append(nn.Linear(self.hid_size, self.num_of_class[-1]))
    

