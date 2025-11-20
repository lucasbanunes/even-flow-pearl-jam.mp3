
"""Small utilities for constructing torch modules by name.

This module provides:

- :data:`ModuleNameType` - a literal type enumerating supported activation
  module names.
- :func:`torch_module_from_string` - a helper that constructs a ``torch.nn``
  module instance from a literal name. The function is used by other builders
  in the project (for example the MLP builder) to convert a configuration
  string into an actual activation module.

The public API is intentionally small and strict: only the activation names
``'relu'``, ``'sigmoid'`` and ``'tanh'`` are supported. Passing an unsupported
name raises ``ValueError``.
"""

from typing import Literal
from torch import nn


type ModuleNameType = Literal['relu', 'sigmoid', 'tanh']


TORCH_LOSSES = {
    'mse': nn.MSELoss
}

TORCH_MODULES = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}
