"""even_flow.models.vector_field
================================

Time-conditional MLP vector field used by neural ODE models.

This module provides :class:`MLPVectorField`, a PyTorch ``nn.Module`` that
augments input state vectors with sinusoidal time embeddings and passes them
through a small MLP to produce a time-conditional vector field (for example,
dx/dt).

The file uses NumPy-style docstrings for public classes and methods.
"""

import torch
import torch.nn as nn

from .mlp import build_mlp, DimsType, ActivationsType
from ..utils import interleave_columns


class MLPVectorField(nn.Module):
    """Time-conditional MLP vector field.

    This module concatenates a sinusoidal time embedding to the input state
    vector and passes the result through a multilayer perceptron (MLP).

    Parameters
    ----------
    input_dims : int
        Number of dimensions of the state vector (per sample).
    time_embed_dims : int
        Size of the sinusoidal time embedding (should be even since sin and
        cos outputs are interleaved).
    time_embed_freq : float
        Base frequency used to build the sinusoidal embeddings. Higher values
        change the scale of the frequencies used.
    neurons_per_layer : sequence of int
        Sequence listing the neuron counts for each hidden (and final) MLP
        layer. The first element of the MLP receives ``input_dims +
        time_embed_dims`` features.
    activations : list of strings
        Activation(s) used between MLP layers. Passed directly to
        :func:`build_mlp`.

    Attributes
    ----------
    input_dims : int
        See parameter description.
    time_embed_dims : int
        See parameter description.
    time_embed_freq : float
        See parameter description.
    model_dims : list[int]
        Internal list of MLP layer sizes used to build the network.
    model : nn.Module
        The MLP implementing the vector field.
    """

    def __init__(self,
                 input_dims: int,
                 time_embed_dims: int,
                 time_embed_freq: float,
                 neurons_per_layer: DimsType,
                 activations: ActivationsType):
        super(MLPVectorField, self).__init__()

        self.input_dims = input_dims
        self.time_embed_dims = time_embed_dims
        self.time_embed_freq = time_embed_freq
        self.model_dims = [self.input_dims +
                           self.time_embed_dims] + neurons_per_layer
        self.activations = activations
        self.model = build_mlp(self.model_dims, self.activations)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embeddings and interleave sin/cos columns.

        The embedding uses linearly spaced frequencies similar to the
        Transformer positional embeddings. For an input ``t`` with shape
        ``(batch_size, 1)`` the method returns a tensor with shape
        ``(batch_size, time_embed_dims)`` where columns alternate between
        ``sin`` and ``cos`` terms for each frequency.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor of shape ``(batch_size, 1)``. Must have a floating
            dtype compatible with the module parameters.

        Returns
        -------
        torch.Tensor
            Time embedding tensor with shape ``(batch_size, time_embed_dims)``
            and the same dtype/device as ``t``.

        Notes
        -----
        The implementation assumes ``time_embed_dims`` is even (so it can be
        split into sin and cos halves). If an odd value is provided, the
        implementation will implicitly floor the half-size and produce an
        embedding of size ``2 * floor(time_embed_dims/2)``.
        """
        half_dim = self.time_embed_dims // 2
        freq_exponents = torch.arange(0, half_dim, dtype=t.dtype) / half_dim
        freqs = 1.0 / (self.time_embed_freq ** freq_exponents)  # Shape: (half_dim,)
        args = t * freqs.view(1, half_dim)  # Shape: (batch_size, half_dim)
        sin = torch.sin(args)
        cos = torch.cos(args)
        return interleave_columns(sin, cos)  # Shape: (batch_size, time_embed_dims)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate state with time embedding and apply MLP.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor with shape ``(batch_size, 1)``. The same ``batch_size``
            as ``x`` is expected.
        x : torch.Tensor
            State tensor with shape ``(batch_size, input_dims)``.

        Returns
        -------
        torch.Tensor
            Output of the MLP; shape is ``(batch_size, output_dim)`` where
            ``output_dim`` equals the last element of ``neurons_per_layer``
            supplied at construction. The dtype/device matches the input
            tensors.
        """
        model_input = torch.cat([x, self.time_embedding(t)], dim=-1)
        return self.model(model_input)
