from dotmap import DotMap

models_implementations_dict = {}


def register_model(name: str = None) -> type:
    """
    This decorator function is used to register (store in a dictionary) classes of implemented model architectures.
    Args:
        name (str): The key of the model class in a dictionary. Can be derived from the class name if no name is
            specified.
    Returns:
        type: Type of the registered class.
    """
    def decorate(cls: type, model_name: str = None) -> type:
        global models_implementations_dict

        if not model_name:
            model_name = cls.__module__ + '.' + cls.__name__
        if model_name in models_implementations_dict:
            print("Model class {} is already registered and will be overwritten!".format(model_name))

        models_implementations_dict[model_name] = cls

        return cls

    return lambda cls: decorate(cls, name)


class ModelBuilder(object):
    """
    This class is a factory for models creation.
    Attributes:
        models_dict (dict): Dictionary with model name as a key and class as a value.
    """
    models_dict = models_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates model object using model type specified in configs.
        Args:
            configs (DotMap): Object contains parameters required for model usage.
        Raises:
            KeyError: No architecture found for the specified model type.
        Returns:
            Model: Created object of model
        """
        try:
            return cls.models_dict[configs.model_type](configs)
        except KeyError:
            raise "{} architecture not implemented!".format(configs.model_type)