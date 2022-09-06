from unittest import TestCase
import torch.nn
from ohlcformer.losses import MaskedRMSELoss
from tests.losses.test_losses_utils import default_test_loss_init, test_loss_forward


class TestRMSELoss(TestCase):
    def test_init(self):
        self.assertTrue(default_test_loss_init(MaskedRMSELoss))

    def test_forward(self):

        class RMSELoss(torch.nn.Module):
            def __call__(self, prediction, target):
                return torch.sqrt(torch.pow(prediction - target, 2)).mean()

        self.assertTrue(test_loss_forward(MaskedRMSELoss, RMSELoss))
