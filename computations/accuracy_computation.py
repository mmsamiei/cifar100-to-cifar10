import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy_computation(model, head_number, x, y):
    """
        Computes the accuracy on a batch of validation inputs
        Args:
            model
            head_number: int
            x: tensor in shape of [batch, 3, 32, 32]
            y: tensor in shape of [batch]

        Returns:
            This is a description of what is returned.
    """
    model.eval()
    #model_output = model(x)[head_number] ## score: [batch, num_class]
    predicted = model(x)[head_number].argmax(dim=1) ## [batch]
    num_correct = (predicted == y).sum().item()
    return {'num_correct': num_correct, 'total': len(x)}