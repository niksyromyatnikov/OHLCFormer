import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.dropout = nn.Dropout(p=configs.dropout_proba)
        self.ff = nn.Linear(configs.input_size, configs.hidden_size, bias=True)

        position = torch.arange(configs.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, configs.hidden_size, 2) * (-math.log(10000.0) / configs.hidden_size))
        pe = torch.zeros(configs.max_seq_length, configs.hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.ff(input_ids)
        x = x + self.pe[:x.size(-2)]
        return self.dropout(x)


class IntervalEncoding(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.dropout = nn.Dropout(p=configs.dropout_proba)
        self.ff = nn.Linear(configs.input_size, configs.hidden_size, bias=True)

        self.encoding_epsilon = configs.encoding_epsilon  # nn.Parameter(torch.tensor([configs.encoding_epsilon]), requires_grad=True)
        self.encoding_gamma = configs.encoding_gamma  # nn.Parameter(torch.tensor([configs.encoding_gamma]), requires_grad=True)
        self.encoding_beta = configs.encoding_beta  # nn.Parameter(torch.tensor([configs.encoding_beta]), requires_grad=True)

        position = torch.arange(configs.max_seq_length).unsqueeze(1)
        x = position * torch.exp(torch.arange(0, configs.hidden_size, 2) * (-math.log(10000) / configs.hidden_size))
        inc_term = x / configs.max_seq_length ** (1 / self.encoding_epsilon) - self.encoding_gamma ** x

        pe = torch.zeros(configs.max_seq_length, configs.hidden_size)
        pe[:, 0::2] = self.normalize(torch.sin(x / self.encoding_beta) + inc_term)
        pe[:, 1::2] = self.normalize(torch.cos(x / self.encoding_beta) + inc_term)
        self.register_buffer('pe', pe)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.ff(input_ids)
        x = x + self.pe[:x.size(-2)]
        return self.dropout(x)

    def normalize(self, var, a=-1, b=1):
        var_min = torch.min(var)
        var_max = torch.max(var)
        return (b - a) * (var - var_min) / (var_max - var_min) + a


class IntervalEncodingTrainable(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.dropout = nn.Dropout(p=configs.dropout_proba)
        self.ff = nn.Linear(configs.input_size, configs.hidden_size, bias=True)

        # self.encoding_epsilon = nn.Parameter(torch.tensor([configs.encoding_epsilon]), requires_grad=True)
        # self.encoding_gamma = nn.Parameter(torch.tensor([configs.encoding_gamma]), requires_grad=True)
        # self.encoding_beta = nn.Parameter(torch.tensor([configs.encoding_beta]), requires_grad=True)

        # position = torch.arange(configs.max_seq_length).unsqueeze(1)

        # x_arg = position * torch.exp(torch.arange(0, configs.hidden_size, 2) * (-math.log(10000) / configs.hidden_size))
        # self.x_arg = nn.Parameter(x_arg, requires_grad=False)

        self.pe = torch.nn.Parameter(torch.FloatTensor(configs.max_seq_length, configs.hidden_size).uniform_(),
                                     requires_grad=True)  # torch.zeros(configs.max_seq_length, configs.hidden_size)

        # self.max_seq_length = nn.Parameter(torch.tensor([configs.max_seq_length]), requires_grad=False)
        # self.inc_term = nn.Parameter(torch.zeros(x_arg.shape), requires_grad=True)

        # self.register_buffer('pe', pe)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # self.inc_term = nn.Parameter(self.x_arg / self.max_seq_length ** (1 / self.encoding_epsilon) - self.encoding_gamma ** self.x_arg)

        # self.pe[:, 0::2] = self.normalize(torch.sin(self.x_arg / self.encoding_beta) + self.inc_term)
        # self.pe[:, 1::2] = self.normalize(torch.cos(self.x_arg / self.encoding_beta) + self.inc_term)

        x = self.ff(input_ids)
        x = x + self.pe[:x.size(-2)]
        return self.dropout(x)

    def normalize(self, var, a=-1, b=1):
        var_min = torch.min(var)
        var_max = torch.max(var)
        return (b - a) * (var - var_min) / (var_max - var_min) + a
