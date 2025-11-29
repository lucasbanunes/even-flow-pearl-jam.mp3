"""MLP helper utilities.

Helpers for constructing small multilayer perceptrons (MLPs) used across the
project. The builder returns a ``torch.nn.Sequential`` model that alternates
``nn.Linear`` layers with activation modules. Activation modules are looked
up from the ``TORCH_MODULES`` mapping imported from ``even_flow.torch`` and
instantiated when requested.

This module focuses on a compact, configuration-driven builder used by the
vector-field and model setup code.
"""

import json
from tempfile import TemporaryDirectory
from typing import Annotated, Any, ClassVar, Self
from pathlib import Path
import torch
import torch.nn as nn
from pydantic import Field
import mlflow

from ..pydantic import MLFlowLoggedModel
from ..torch import TORCH_MODULES
from ..utils import interleave_columns
from ..mlflow import load_json as mlflow_load_json


# Type aliases for readability
type DimsType = Annotated[
    list[int],
    Field(
        min_items=2,
        description="List of layer dimensions; must have at least one entry."
    )
]
type ActivationsType = Annotated[
    list[str | None],
    Field(
        min_items=1,
        description="List of activation names length should be len(dims)-1."
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
        if activation is None or activation == 'linear':
            continue
        model.append(TORCH_MODULES[activation]())
    model.example_input_array = torch.randn(dims[0])
    return model


type InputDimsType = Annotated[
    int,
    Field(
        description="Dimensionality of the input data."
    )
]

type TimeEmbedDimsType = Annotated[
    int,
    Field(
        description="Dimensionality of the time embedding."
    )
]

type TimeEmbedFreqType = Annotated[
    float,
    Field(
        description="Base frequency for the time embedding."
    )
]


class TimeEmbeddingMLP(nn.Module):

    def __init__(self,
                 input_dims: InputDimsType,
                 time_embed_dims: TimeEmbedDimsType,
                 time_embed_freq: TimeEmbedFreqType,
                 neurons_per_layer: DimsType,
                 activations: ActivationsType):
        super().__init__()

        self.input_dims = input_dims
        self.time_embed_dims = time_embed_dims
        self.time_embed_freq = time_embed_freq
        self.model_input_dims = (self.input_dims +
                                 2 * (self.time_embed_dims // 2))
        self.model_dims = [self.model_input_dims] + neurons_per_layer
        self.output_dims = self.model_dims[-1]
        self.activations = activations
        self.model = build_mlp(self.model_dims, self.activations)
        self.nfe = 0

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_embed_dims // 2
        freq_exponents = torch.arange(0, half_dim, dtype=t.dtype) / half_dim
        # Shape: (half_dim,)
        freqs = 1.0 / (self.time_embed_freq ** freq_exponents)
        args = t * freqs.view(1, half_dim)  # Shape: (batch_size, half_dim)
        sin = torch.sin(args)
        cos = torch.cos(args)
        # Shape: (batch_size, time_embed_dims)
        return interleave_columns(sin, cos)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0], 1), t.float().item())
        model_input = torch.cat([x, self.time_embedding(t)], dim=-1)
        self.nfe += 1
        return self.model(model_input)

    @classmethod
    def pydantic_before_validator(cls, v: Any) -> Self:
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Cannot convert {type(v)} to {cls}.")

    @staticmethod
    def pydantic_plain_serializer(v: 'TimeEmbeddingMLP') -> dict[str, Any]:
        return {
            "input_dims": v.input_dims,
            "time_embed_dims": v.time_embed_dims,
            "time_embed_freq": v.time_embed_freq,
            "model_dims": v.model_dims,
            "activations": v.activations,
        }

    def reset_metrics(self) -> None:
        self.nfe = 0


class TimeEmbeddingMLPConfig(MLFlowLoggedModel):

    JSON_ARTIFACT_PATH: ClassVar[str] = 'time_embedding_mlp_config.json'

    input_dims: InputDimsType
    time_embed_dims: TimeEmbedDimsType
    time_embed_freq: TimeEmbedFreqType
    neurons_per_layer: DimsType
    activations: ActivationsType

    def as_nn_module(self) -> TimeEmbeddingMLP:
        return TimeEmbeddingMLP(
            input_dims=self.input_dims,
            time_embed_dims=self.time_embed_dims,
            time_embed_freq=self.time_embed_freq,
            neurons_per_layer=self.neurons_per_layer,
            activations=self.activations
        )

    @classmethod
    def from_mlflow(cls, mlflow_run, prefix=''):
        if prefix:
            prefix = prefix.replace('.', '_') + '_'
        artifact_name = f'{prefix}{cls.JSON_ARTIFACT_PATH}'
        config_dict = mlflow_load_json(
            run_id=mlflow_run.info.run_id,
            artifact_path=artifact_name
        )
        instance = cls(**config_dict)
        return instance

    def _to_mlflow(self, prefix=''):
        if prefix:
            file_prefix = prefix.replace('.', '_') + '_'
            prefix += '.'
        mlflow.log_param(f'{prefix}input_dims', self.input_dims)
        mlflow.log_param(f'{prefix}time_embed_dims', self.time_embed_dims)
        mlflow.log_param(f'{prefix}time_embed_freq', self.time_embed_freq)
        mlflow.log_param(f'{prefix}neurons_per_layer',
                         json.dumps(self.neurons_per_layer))
        mlflow.log_param(f'{prefix}activations', json.dumps(self.activations))
        json_str = self.model_dump_json(indent=4,
                                        exclude=['id_', 'name'])
        filename = file_prefix + self.JSON_ARTIFACT_PATH
        with TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / filename
            filepath.write_text(json_str)
            mlflow.log_artifact(str(filepath))
