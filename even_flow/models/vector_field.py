import torch
import torch.nn as nn

from .mlp import build_mlp, DimsType, ActivationsType
from ..utils import interleave_columns


class MLPVectorField(nn.Module):
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

    def time_embedding(self, t: torch.Tensor):
        """
        Add sinusoidal time embeddings to the input time tensor.
        Assumes data in (batch_size, 1) shape.

        Parameters
        ----------
        t : torch.Tensor
            Time Tensor of shape (batch_size, 1)
        """
        half_dim = self.time_embed_dims // 2
        freq_exponents = torch.arange(0, half_dim, dtype=t.dtype) / half_dim
        freqs = 1.0 / (self.time_embed_freq ** freq_exponents)  # Shape: (half_dim,)
        args = t * freqs.view(1, half_dim)  # Shape: (batch_size, half_dim)
        sin = torch.sin(args)
        cos = torch.cos(args)
        return interleave_columns(sin, cos)  # Shape: (batch_size, time_embed_dims)

    def forward(self, x, t):
        model_input = torch.cat([x, self.time_embedding(t)], dim=-1)
        return self.model(model_input)
