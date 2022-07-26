import math
import torch
from torch import nn
from . import Encoding


class PositionalEncoding(Encoding):

    def __init__(self, configs):
        super(PositionalEncoding, self).__init__(configs)
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
            input_ids: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.ff(input_ids)
        x = x + self.pe[:x.size(-2)]
        return self.dropout(x)