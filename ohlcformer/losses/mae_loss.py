import torch
from .builder import register_loss
from .loss import Loss


@register_loss('masked_mae_loss')
class MaskedMAELoss(Loss):
    def __init__(self, weighted=False, reduction='sum', name=None):
        super(MaskedMAELoss, self).__init__(weighted, reduction, name)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        reduction = reduction if reduction else self.reduction

        out = None
        lens = torch.count_nonzero(mask != ignored_index, axis=1)

        for i in range(0, target.shape[0]):
            diff = torch.abs(prediction[i][mask[i].ravel() != ignored_index] - target[i][:lens[i]])

            if out is None:
                out = diff
            else:
                out = torch.cat([out, diff])

        if reduction == "mean":
            return out.mean()
        if reduction == "sum":
            return out.sum()

        return out
