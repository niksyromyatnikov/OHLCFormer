from unittest import TestCase
from dotmap import DotMap
from ohlcformer.losses import Loss
from ohlcformer.losses import builder


class TestLossBuilder(TestCase):
    def test_register_loss(self):
        loss_name = 'test_loss'

        @builder.register_loss(loss_name)
        class TestLoss:
            pass

        self.assertEqual(TestLoss, builder.losses_implementations_dict[loss_name])

        @builder.register_loss()
        class TestLoss2:
            pass

        loss_name = TestLoss2.__module__ + '.' + TestLoss2.__name__
        self.assertEqual(TestLoss2, builder.losses_implementations_dict[loss_name])

    def test_loss_builder(self):
        LossBuilder = builder.LossBuilder()

        for _, loss_type in LossBuilder.losses_dict.items():
            self.assertEqual(issubclass(loss_type, Loss), True)

        loss_configs = DotMap({"loss_type": "masked_mse_loss", "weighted": False, "reduction": "sum", "name": "loss"})
        loss = LossBuilder.build(loss_configs)
        self.assertEqual(isinstance(loss, Loss), True)
        self.assertEqual(loss.name, loss_configs.name)

        loss_configs.loss_type = "not_implemented_loss"
        self.assertRaises(KeyError, LossBuilder.build, loss_configs)

        loss_configs.loss_type = []
        self.assertRaises(TypeError, LossBuilder.build, loss_configs)

        loss_configs = None
        self.assertRaises(AttributeError, LossBuilder.build, loss_configs)
