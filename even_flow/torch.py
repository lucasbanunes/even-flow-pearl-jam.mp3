
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

from typing import Callable, Literal
import torch
from torch import nn
from torch.func import jvp


type ModuleNameType = Literal['relu', 'sigmoid', 'tanh']


TORCH_LOSSES = {
    'mse': nn.MSELoss
}

TORCH_MODULES = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
}


def memory_optimized_divergence1d(func: Callable[[torch.Tensor], torch.Tensor],
                                  ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Create a divergence computation function using memory-optimized JVP."""
    def compute_divergence(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        divergence = 0
        for i in range(len(x)):
            e_i = torch.zeros_like(x)
            e_i[i] = 1.0
            x_eval, jvp_result = jvp(func, (x,), (e_i,))
            divergence += jvp_result.flatten()[i]
        return x_eval, divergence

    return compute_divergence
