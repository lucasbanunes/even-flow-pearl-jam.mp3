"""MLP helper utilities.

This module provides a small helper to build a multilayer perceptron (MLP)
expressed as a ``torch.nn.Sequential`` module. The builder is intentionally
minimal and expects layer sizes and activation specifiers.

The project uses :func:`build_mlp` when wiring small feed-forward networks for
vector fields and other lightweight components.
"""

import torch
import torch.nn as nn
from . import torch_module_from_string, ModuleNameType


# Type aliases for readability
type DimsType = list[int]
type ActivationsType = list[ModuleNameType | None]


def build_mlp(
    dims: DimsType,
    activations: ActivationsType
) -> nn.Sequential:
    """Build a simple MLP as an ``nn.Sequential``.

    The function constructs a sequential model by creating a linear layer for
    each consecutive pair of dimensions in ``dims`` and optionally inserting
    activation modules between linear layers. Activation modules are created
    by calling :func:`torch_module_from_string` with the provided activation
    name.

    Parameters
    ----------
    dims : list[int]
        Sequence of layer sizes. For example ``[in_dim, hidden, out_dim]``.
    activations : list[str or None]
        Sequence of activation specifiers with the same length as ``len(dims)-1``.
        Each activation may be a module name string accepted by
        :func:`torch_module_from_string` or ``None`` to insert no activation
        after the corresponding linear layer.

    Returns
    -------
    torch.nn.Sequential
        A sequential model that alternates ``nn.Linear`` layers with the
        optional activation modules. Additionally, the returned module will
        have an attribute ``example_input_array`` set to ``torch.randn(dims[0])``
        (useful for tracing or model inspection in some tooling).

    Notes
    -----
    - The function does not validate that ``len(activations) == len(dims)-1``;
      the code will implicitly zip and ignore extra entries if lengths differ.
    - The ``example_input_array`` shape is currently a 1-D tensor; tooling
      that expects a batch dimension may need to wrap it (e.g. ``unsqueeze(0)``).

    Examples
    --------
    >>> model = build_mlp([4, 16, 8], ['relu', None])
    >>> isinstance(model, nn.Sequential)
    True
    """
    model = nn.Sequential()
    iterator = zip(dims[:-1],
                   dims[1:],
                   activations)
    for input_dim, output_dim, activation in iterator:
        model.append(nn.Linear(input_dim, output_dim))
        if activation is not None:
            model.append(torch_module_from_string(activation))
    model.example_input_array = torch.randn(dims[0])
    return model
