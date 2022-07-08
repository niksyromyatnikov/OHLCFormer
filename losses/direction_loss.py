import torch
from sklearn.metrics import f1_score

from losses.loss import Loss


class MaskedDirectionLoss(Loss):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedDirectionLoss, self).__init__(weighted, reduction)
        self.criterion = torch.nn.BCELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None,
                return_f1=False):
        reduction = reduction if reduction else self.reduction

        out = None
        prediction_flat = None
        target_flat = None

        lens = torch.count_nonzero(mask, axis=1)

        for i in range(0, target.shape[0]):
            if prediction_flat is None:
                prediction_flat = prediction[i][mask[i].ravel() != ignored_index].ravel()
            else:
                prediction_flat = torch.cat([prediction_flat, prediction[i][mask[i].ravel() != ignored_index].ravel()])

            if target_flat is None:
                target_flat = target[i][:lens[i]].ravel()
            else:
                target_flat = torch.cat([target_flat, target[i][:lens[i]].ravel()])

        prediction_labeled = torch.signbit(prediction_flat)
        target_labeled = torch.signbit(target_flat)

        loss = self.criterion(prediction_labeled.float(), target_labeled.float())

        if reduction == "mean":
            out = loss.mean()
        elif reduction == "sum":
            out = loss.sum()

        if return_f1:
            return out, f1_score(prediction_labeled.cpu(), target_labeled.cpu())

        return out
