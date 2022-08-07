import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, weighted=False, reduction='sum', name=None):
        super(Loss, self).__init__()
        self.weighted = weighted
        self.reduction = reduction
        self.name = name if name is not None else self.__class__.__name__

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        pass


def get_metric_direction(metric_name) -> bool:
    if metric_name == "f1":
        return True
    return False
