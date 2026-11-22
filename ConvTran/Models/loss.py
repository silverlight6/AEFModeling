import torch
import torch.nn as nn


def get_loss_module():
    return nn.CrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))
