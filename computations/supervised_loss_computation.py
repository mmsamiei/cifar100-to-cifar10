import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import cross_entropy_loss

def supervised_loss_computation(model, head_number, x, y):
    """
        Computes the loss on a batch of training inputs
        Args:
            model
            head_number: int
            x: tensor in shape of [batch, 3, 32, 32]
            y: tensor in shape of [batch]

        Returns:
            This is a description of what is returned.
    """

    model_output = model(x)[head_number] ## score: [batch, num_class]
    criterion = cross_entropy_loss.CrossEntropyLoss()
    loss = criterion(model_output, y)
    return loss
    
    
