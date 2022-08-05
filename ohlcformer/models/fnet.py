from .builder import register_model
from .model import ModelForFM


@register_model('FNetForFM')
class FNetForFM(ModelForFM):

    def __init__(self, configs):
        super(FNetForFM, self).__init__(configs)

