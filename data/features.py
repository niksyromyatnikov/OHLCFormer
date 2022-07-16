import random
import torch
from torch.utils.data import TensorDataset


class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    def __init__(self, input_ids, attention_mask, mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mask = mask
        self.labels = labels


def convert_to_tensor_dataset(dataset: list,
                              max_seq_len: int = 2000,
                              mask_proba: float = 0.2,
                              prediction_len: int = 5) -> TensorDataset:
    features = []
    labels_pad_len = int(max_seq_len * (mask_proba + 0.05)) if mask_proba > 0 else 0

    for row in dataset:
        attention_mask = [1] * max_seq_len
        mask = [0] * max_seq_len
        labels = []

        seq_len = len(row)
        if seq_len < prediction_len + 0.3 * seq_len:
            continue

        pad_len = max_seq_len - seq_len
        attention_mask[seq_len:] = [0] * pad_len
        vals = [True, False]
        weights = [1 - mask_proba, mask_proba]

        for i, elem in enumerate(row):
            if i >= seq_len - prediction_len or not random.choices(vals, weights)[0]:
                labels.append(elem)
                row[i] = [0] * len(elem)
                mask[i] = 1

        labels = labels + [[0] * 4] * (labels_pad_len - len(labels) if labels_pad_len - len(labels) > 0 else 0)
        input_ids = row + [[0] * len(row[0])] * pad_len
        features.append(InputFeatures(input_ids, attention_mask, mask, labels))

    return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.float),
                         torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                         torch.tensor([f.mask for f in features], dtype=torch.long),
                         torch.tensor([f.labels for f in features], dtype=torch.float))


def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, seq_len: int = 2000, mask_proba: float = 0.2):
    batch_size = input_ids.shape[0]
    element_len = input_ids.shape[-1]

    mask = torch.rand(batch_size, seq_len) < mask_proba
    mask[attention_mask == 0] = 0
    labels_pad_len = torch.max(torch.sum(mask, 1)).item()

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = torch.zeros(1, element_len)

    labels_batch = torch.zeros(batch_size, labels_pad_len, element_len)

    for i, row in enumerate(input_ids):
        labels = row[mask[i]]
        labels_batch[i][:labels.shape[0]] = labels

    return masked_input_ids, labels_batch, mask.long()
