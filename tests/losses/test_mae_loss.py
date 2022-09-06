from unittest import TestCase
import torch.nn
from ohlcformer.losses import MaskedMAELoss
from tests.losses.test_losses_utils import default_test_loss_init, test_loss_forward


class TestMAELoss(TestCase):
    def test_init(self):
        self.assertTrue(default_test_loss_init(MaskedMAELoss))

    def test_forward(self):

        class MAELoss(torch.nn.Module):
            def __call__(self, prediction, target):
                return torch.abs(prediction - target).mean()

        self.assertTrue(test_loss_forward(MaskedMAELoss, MAELoss))
