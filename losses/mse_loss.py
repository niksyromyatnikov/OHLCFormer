import torch
from losses.loss import Loss


class MaskedMSELoss(Loss):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedMSELoss, self).__init__(weighted, reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        reduction = reduction if reduction else self.reduction

        out = None
        lens = torch.count_nonzero(mask, axis=1)

        for i in range(0, target.shape[0]):
            diff = (prediction[i][mask[i].ravel() != ignored_index] - target[i][:lens[i]]) ** 2

            if out is None:
                out = diff
            else:
                out = torch.cat([out, diff])

        if reduction == "mean":
            return out.mean()
        if reduction == "sum":
            return out.sum()

        return out