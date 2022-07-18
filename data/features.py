import random
import torch
from torch.utils.data import TensorDataset


class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    def __init__(self, input_ids, attention_mask, mask=None, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mask = mask
        self.labels = labels


def convert_to_tensor_dataset(dataset: list,
                              max_seq_len: int = 2000,
                              mask_proba: float = 0.2,
                              prediction_len: int = 5,
                              perform_masking: bool = True
                              ) -> TensorDataset:
    features = []
    labels_pad_len = int(max_seq_len * (mask_proba + 0.05)) if mask_proba > 0 else 0

    for row in dataset:
        seq_len = len(row)
        pad_len = max_seq_len - seq_len

        attention_mask = [1] * max_seq_len
        input_ids = row + [[0] * len(row[0])] * pad_len

        if perform_masking:
            mask = [0] * max_seq_len
            labels = []

            if seq_len < prediction_len + 0.3 * seq_len:
                continue

            attention_mask[seq_len:] = [0] * pad_len
            vals = [True, False]
            weights = [1 - mask_proba, mask_proba]

            for i, elem in enumerate(row):
                if i >= seq_len - prediction_len or not random.choices(vals, weights)[0]:
                    labels.append(elem)
                    row[i] = [0] * len(elem)
                    mask[i] = 1

            labels = labels + [[0] * 4] * (labels_pad_len - len(labels) if labels_pad_len - len(labels) > 0 else 0)
            features.append(InputFeatures(input_ids, attention_mask, mask, labels))
        else:
            features.append(InputFeatures(input_ids, attention_mask))

    if perform_masking:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.float),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.mask for f in features], dtype=torch.long),
                             torch.tensor([f.labels for f in features], dtype=torch.float))
    else:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.float),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long))


def mask_tokens(input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                seq_len: int = 2000,
                mask_proba: float = 0.2,
                prediction_len: int = 5
                ) -> tuple:
    batch_size = input_ids.shape[0]
    element_len = input_ids.shape[-1]
    masked_input_ids = input_ids.clone()

    mask = torch.rand(batch_size, seq_len, device=input_ids.device) < mask_proba
    mask[attention_mask == 0] = 0

    labels_pad_len = torch.max(torch.sum(mask, 1)).item() + prediction_len
    labels_batch = torch.zeros(batch_size, labels_pad_len, element_len, device=input_ids.device)

    end_of_seq = torch.argmin(attention_mask, 1)
    end_of_seq[(attention_mask[:, 0] == 1) & (end_of_seq == 0)] = seq_len

    for i, row in enumerate(input_ids):
        mask[i][end_of_seq[i]-prediction_len:end_of_seq[i]] = 1
        labels = row[mask[i]]
        labels_batch[i][:labels.shape[0]] = labels

    masked_input_ids[mask] = torch.zeros(1, element_len, device=input_ids.device)

    return masked_input_ids, labels_batch, mask.long()
