
from typing import Literal
from torch import nn


type ModuleNameType = Literal['relu', 'sigmoid', 'tanh']


def torch_module_from_string(name: ModuleNameType, **kwargs):
    match name:
        case 'relu':
            return nn.ReLU(**kwargs)
        case 'sigmoid':
            return nn.Sigmoid(**kwargs)
        case 'tanh':
            return nn.Tanh(**kwargs)
        case _:
            raise ValueError(
                f'Unsupported module: {name}')
