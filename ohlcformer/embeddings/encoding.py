import torch
from torch import nn


class Encoding(nn.Module):

    def __init__(self, configs):
        super(Encoding, self).__init__()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pass

    def normalize(self, var, a=-1, b=1):
        var_min = torch.min(var)
        var_max = torch.max(var)
        return (b - a) * (var - var_min) / (var_max - var_min) + a
