from unittest import TestCase
from tests.losses.test_losses_utils import default_test_loss_init


class TestLoss(TestCase):
    def test_init(self):
        self.assertTrue(default_test_loss_init())
