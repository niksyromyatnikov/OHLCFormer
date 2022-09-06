from unittest import TestCase
import torch
from ohlcformer.losses import MaskedDirectionLoss
from tests.losses.test_losses_utils import default_test_loss_init, test_loss_forward


class TestDirectionLoss(TestCase):
    def test_init(self):
        self.assertTrue(default_test_loss_init(MaskedDirectionLoss))

    def test_forward(self):
        self.assertTrue(test_loss_forward(MaskedDirectionLoss, torch.nn.BCELoss, test_f1=True))
