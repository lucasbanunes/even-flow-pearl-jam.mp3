
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


def torch_module_from_string(name: ModuleNameType, **kwargs) -> nn.Module:
    """Return a torch activation module given its string name.

    Parameters
    ----------
    name : {'relu', 'sigmoid', 'tanh'}
        Name of the activation module to construct. Must be one of the values
        enumerated by :data:`ModuleNameType`.
    **kwargs
        Keyword arguments forwarded to the activation class constructor. For
        most simple activations (ReLU, Sigmoid, Tanh) there are no required
        kwargs, but this allows future extensibility.

    Returns
    -------
    torch.nn.Module
        An instance of the requested activation module (for example
        ``nn.ReLU()``, ``nn.Sigmoid()``, ``nn.Tanh()``).

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported module names.

    Examples
    --------
    >>> torch_module_from_string('relu')
    ReLU()

    """
    match name:
        case 'relu':
            return nn.ReLU(**kwargs)
        case 'sigmoid':
            return nn.Sigmoid(**kwargs)
        case 'tanh':
            return nn.Tanh(**kwargs)
        case _:
            raise ValueError(f'Unsupported module: {name}')
