from unittest import TestCase
import torch.nn
from ohlcformer.losses import MaskedMSELoss
from tests.losses.test_losses_utils import default_test_loss_init, test_loss_forward


class TestMSELoss(TestCase):
    def test_init(self):
        self.assertTrue(default_test_loss_init(MaskedMSELoss))

    def test_forward(self):

        class MSELoss(torch.nn.Module):
            def __call__(self, prediction, target):
                return torch.pow(prediction - target, 2).mean()

        self.assertTrue(test_loss_forward(MaskedMSELoss, MSELoss))
