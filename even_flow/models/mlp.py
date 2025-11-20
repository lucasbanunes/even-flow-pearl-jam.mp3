"""MLP helper utilities.

Helpers for constructing small multilayer perceptrons (MLPs) used across the
project. The builder returns a ``torch.nn.Sequential`` model that alternates
``nn.Linear`` layers with activation modules. Activation modules are looked
up from the ``TORCH_MODULES`` mapping imported from ``even_flow.torch`` and
instantiated when requested.

This module focuses on a compact, configuration-driven builder used by the
vector-field and model setup code.
"""

from typing import Annotated
import torch
import torch.nn as nn
from pydantic import Field
from ..torch import TORCH_MODULES


# Type aliases for readability
type DimsType = Annotated[
    list[int],
    Field(
        min_items=2,
        help="List of layer dimensions; must have at least one entry."
    )
]
type ActivationsType = Annotated[
    list[str | None],
    Field(
        min_items=1,
        help="List of activation names length should be len(dims)-1."
    )
]


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
        Sequence of activation specifiers with the same length as
        ``len(dims)-1``. Each entry should be either ``None`` (no activation)
        or a string key present in the ``TORCH_MODULES`` mapping imported at
        the top of this module. For example ``['relu', None, 'tanh']`` will
        insert a ReLU after the first linear layer, no activation after the
        second, and a Tanh after the third.

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
        - Activation strings are resolved via ``TORCH_MODULES[activation]`` and
            the resulting callable is instantiated without arguments. If the
            mapping expects constructor arguments, update the call sites accordingly.
        - The ``example_input_array`` attribute is set to a 1-D tensor; tooling
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
            model.append(TORCH_MODULES[activation]())
    model.example_input_array = torch.randn(dims[0])
    return model
