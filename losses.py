import torch
from sklearn.metrics import f1_score
from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)
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


class MaskedRMSELoss(nn.Module):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedRMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)
        reduction = reduction if reduction else self.reduction

        out = None
        lens = torch.count_nonzero(mask, axis=1)

        for i in range(0, target.shape[0]):
            diff = torch.sqrt((prediction[i][mask[i].ravel() != ignored_index] - target[i][:lens[i]]) ** 2)

            if out is None:
                out = diff
            else:
                out = torch.cat([out, diff])

        if reduction == "mean":
            return out.mean()
        if reduction == "sum":
            return out.sum()

        return out


class MaskedMAELoss(torch.nn.Module):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedMAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)
        reduction = reduction if reduction else self.reduction

        out = None
        lens = torch.count_nonzero(mask, axis=1)

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

class MaskedDirectionLoss(nn.Module):
    def __init__(self, weighted=False, reduction='sum'):
        super(MaskedDirectionLoss, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction=None,
                return_f1=False):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)
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


"""
class MaskedMSELoss(nn.Module):
    def __init__(self, weighted=False):
        super(MaskedMSELoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction='None'):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)

        out = (prediction[mask != ignored_index] - target.view(-1, prediction.shape[-1])) ** 2

        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out.sum()


class MaskedRMSELoss(nn.Module):
    def __init__(self, weighted=False):
        super(MaskedRMSELoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction='None'):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)

        out = torch.sqrt((prediction[mask != ignored_index] - target.view(-1, prediction.shape[-1])) ** 2)

        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out.sum()


class MaskedMAELoss(nn.Module):
    def __init__(self, weighted=False):
        super(MaskedMAELoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask, ignored_index=0, reduction='None'):
        # print(prediction, prediction.shape)
        # print(target, target.shape)
        # print(mask, mask.shape)

        out = torch.abs(prediction[mask != ignored_index] - target.view(-1, prediction.shape[-1]))

        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out.sum()
"""