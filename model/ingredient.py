from .neural_net import TwoLayerNet
from sacred import Ingredient


model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    name='nn'
    input_size  = 32 * 32 * 3
    hidden_size = 128
    num_classes = 10


@model_ingredient.named_config
def model500():
    hidden_size = 500


@model_ingredient.named_config
def model600():
    hidden_size = 600


@model_ingredient.capture
def get_model(input_size, hidden_size, num_classes):
    return TwoLayerNet(input_size, hidden_size, num_classes)
