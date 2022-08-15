import copy
from dotmap import DotMap
from ohlcformer import logging

logger = logging.get_logger(__name__)

losses_implementations_dict = {}


def register_loss(name: str = None) -> type:
    """
    This decorator function is used to register (store in a dictionary) classes of implemented loss architectures.
    Args:
        name (str): The key of the loss class in a dictionary. Can be derived from the class name if no name is
            specified.
    Returns:
        type: Type of the registered class.
    """

    def decorate(cls: type, loss_name: str = None) -> type:
        global losses_implementations_dict

        if not loss_name:
            loss_name = cls.__module__ + '.' + cls.__name__
        if loss_name in losses_implementations_dict:
            logger.warning(f"Loss class {loss_name} is already registered and will be overwritten!")

        losses_implementations_dict[loss_name] = cls

        return cls

    return lambda cls: decorate(cls, name)


class LossBuilder(object):
    """
    This class is a factory for losses creation.
    Attributes:
        losses_dict (dict): Dictionary with loss name as a key and class as a value.
    """
    losses_dict = losses_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates loss object using loss type specified in configs.
        Args:
            configs (DotMap): Object contains parameters required for loss usage.
        Raises:
            KeyError: No architecture found for the specified loss type.
        Returns:
            Loss: Created object of loss
        """
        conf = copy.deepcopy(configs)
        loss_type = conf.pop('loss_type')
        try:
            return cls.losses_dict[loss_type](**conf)
        except KeyError:
            logger.error(f'No implementation found for the specified loss type {loss_type}.')
            raise "{} loss not implemented!".format(loss_type)
        except TypeError:
            logger.error(f'Incorrect loss type {type(loss_type)}.')
            raise TypeError("Loss type is not specified!")
