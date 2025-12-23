"""MLP helper utilities.

Helpers for constructing small multilayer perceptrons (MLPs) used across the
project. The builder returns a ``torch.nn.Sequential`` model that alternates
``nn.Linear`` layers with activation modules. Activation modules are looked
up from the ``TORCH_MODULES`` mapping imported from ``even_flow.torch`` and
instantiated when requested.

This module focuses on a compact, configuration-driven builder used by the
vector-field and model setup code.
"""
import math
import json
from tempfile import TemporaryDirectory
from typing import Annotated, Any, ClassVar
from pathlib import Path
import torch
import torch.nn as nn
from pydantic import Field
import mlflow

from ..pydantic import MLFlowLoggedModel
from ..torch import TORCH_MODULES
from ..mlflow import load_json as mlflow_load_json


# Type aliases for readability
type DimsType = Annotated[
    list[int],
    Field(
        min_length=2,
        description="List of layer dimensions. must have at least one entry."
    )
]
type ActivationsType = Annotated[
    list[str | None],
    Field(
        min_length=1,
        description="List of activation names length should be len(dims)-1."
    )
]


class MLP(nn.Sequential):
    """Simple MLP constructed via ``build_mlp``.

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
    """

    def __init__(
        self,
        dims: DimsType,
        activations: ActivationsType
    ):
        super().__init__()
        iterator = zip(dims[:-1],
                       dims[1:],
                       activations)
        for input_dim, output_dim, activation in iterator:
            self.append(nn.Linear(input_dim, output_dim))
            if activation is None or activation == 'linear':
                continue
            self.append(TORCH_MODULES[activation]())
        self.example_input_array = torch.randn(dims[0])


type InputDimsType = Annotated[
    int,
    Field(
        description="Dimensionality of the input data."
    )
]

type TimeEmbedFreqType = Annotated[
    int,
    Field(
        description="Number of frequencies for time embedding."
    )
]


class TimeEmbeddingMLP(MLP):

    def __init__(self,
                 model_config: 'TimeEmbeddingMLPConfig'):
        self.input_dim = model_config.input_dim
        self.real_dims = model_config.input_dim + 2 * model_config.freqs
        self.freqs = model_config.freqs
        neurons = [self.real_dims] + model_config.neurons_per_layer
        super().__init__(
            dims=neurons,
            activations=model_config.activations
        )
        pi = torch.tensor(math.pi)
        self.register_buffer("freqs_array", torch.linspace(
            0, 2*pi, steps=model_config.freqs, dtype=torch.float32))
        self.nfe = torch.tensor(0)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0], 1), t.float().to(x.dtype).item(), device=x.device)
        t = self.freqs_array * t
        model_input = torch.cat([t.cos(), t.sin(), x], dim=-1)
        self.nfe += 1
        return super().forward(model_input)

    def reset_metrics(self) -> None:
        self.nfe = 0


class TimeEmbeddingMLPConfig(MLFlowLoggedModel):

    JSON_ARTIFACT_PATH: ClassVar[str] = 'time_embedding_mlp_config.json'

    input_dim: InputDimsType
    freqs: TimeEmbedFreqType
    neurons_per_layer: DimsType
    activations: ActivationsType

    def as_nn_module(self) -> TimeEmbeddingMLP:
        return TimeEmbeddingMLP(self)

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        if prefix:
            prefix = prefix.replace('.', '_') + '_'
        artifact_name = f'{prefix}{cls.JSON_ARTIFACT_PATH}'
        config_dict = mlflow_load_json(
            run_id=mlflow_run.info.run_id,
            artifact_path=artifact_name
        )
        kwargs.update(config_dict)
        return kwargs

    def _to_mlflow(self, prefix=''):
        if prefix:
            file_prefix = prefix.replace('.', '_') + '_'
            prefix += '.'
        mlflow.log_param(f'{prefix}input_dim', self.input_dim)
        mlflow.log_param(f'{prefix}freqs', self.freqs)
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
