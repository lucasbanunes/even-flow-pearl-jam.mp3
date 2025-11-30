from typing import Annotated, ClassVar, Type, Literal

import mlflow
from pydantic import Field, PrivateAttr
import torch
from torchdiffeq import odeint, odeint_adjoint
import lightning as L
import json
from mlflow.entities import Run

from torchmetrics import MetricCollection, MeanMetric

from .lightning import (
    LightningModel,
    LearningRateType
)
from .mlp import TimeEmbeddingMLPConfig
from ..pydantic import MLFlowLoggedModel


type AdjointType = Annotated[
    bool,
    Field(
        description="Whether to use the adjoint method for gradient computation."
    )
]

type SolverType = Annotated[
    str,
    Field(
        description="The ODE solver method passed to `odeint`/`odeint_adjoint`."
    )
]

type AToleranceType = Annotated[
    float,
    Field(
        gt=0.0,
        description="Absolute Tolerance for the ODE solver."
    )
]

type RToleranceType = Annotated[
    float,
    Field(
        gt=0.0,
        description="Relative Tolerance for the ODE solver."
    )
]

type BaseDistributionType = Annotated[
    Literal['standard_normal', 'uniform'],
    Field(
        description="The base distribution for the CNF."
    )
]

type IntegrationTimesType = Annotated[
    list[float] | None,
    Field(None, description="The timespan for integration")
]


class NeuralODEModule(L.LightningModule):

    def __init__(self,
                 model_config: 'NeuralODEModel'):

        super().__init__()
        super().save_hyperparameters(logger=False)
        self.vector_field = model_config.vector_field.as_nn_module()
        self.solver = model_config.solver
        self.atol = model_config.atol
        self.rtol = model_config.rtol
        self.adjoint = model_config.adjoint
        if self.adjoint:
            self.odeint_func = odeint_adjoint
        else:
            self.odeint_func = odeint

        self.input_shape = model_config.input_shape
        self.sum_dims = tuple(range(1, len(self.input_shape)+1))
        base_distribution = model_config.base_distribution
        if base_distribution is None or base_distribution == 'standard_normal':
            self.base_distribution = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.input_shape),
                covariance_matrix=torch.eye(self.input_shape[0])
            )
        elif base_distribution == 'uniform':
            self.base_distribution = torch.distributions.Uniform(
                low=-1.0, high=1.0)
        else:
            raise ValueError(
                f"Unsupported base distribution: {base_distribution}")

        self.learning_rate = model_config.learning_rate
        self.integration_times = torch.Tensor(
            model_config.integration_times).float()

        self.train_metrics = MetricCollection({
            'loss': MeanMetric()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.odeint_func(
            func=self.vector_field,
            y0=x,
            t=self.integration_times,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        return self.base_distribution.log_prob(z[-1]).reshape(-1, 1)

    @torch.no_grad()
    def sample(self, shape: tuple[int]) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.base_distribution.sample(shape)
        x = odeint(
            func=self.vector_field,
            y0=z,
            t=self.integration_times.flip(0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        return z, x[-1]

    @torch.no_grad()
    def trajectory(self,
                   x: torch.Tensor,
                   integration_times: torch.Tensor) -> torch.Tensor:
        traj = odeint(
            func=self.vector_field,
            y0=x,
            t=integration_times,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        return traj

    def training_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.train_metrics.update(loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.val_metrics.update(loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.test_metrics.update(loss)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        metric_values = self.test_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"test_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.vector_field.reset_metrics()

    def get_test_metrics(self) -> dict[str, float]:
        metrics = self.test_metrics.compute()
        metrics['nfe'] = self.vector_field.nfe
        return metrics

    def get_val_metrics(self) -> dict[str, float]:
        return self.val_metrics.compute()

    def get_train_metrics(self) -> dict[str, float]:
        return self.train_metrics.compute()


class NeuralODEModel(LightningModel):

    BASE_PROFILER_FILENAME: ClassVar[str] = 'profiler'
    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'neuralode.ckpt'
    LIGHTNING_MODULE_TYPE: ClassVar[Type[L.LightningModule]] = NeuralODEModule

    vector_field: MLFlowLoggedModel
    base_distribution: BaseDistributionType = 'standard_normal'
    integration_times: IntegrationTimesType = [0.0, 1.0]
    learning_rate: LearningRateType = 1e-3
    adjoint: AdjointType = True
    solver: SolverType = 'dopri5'
    atol: AToleranceType = 1e-6
    rtol: RToleranceType = 1e-6
    input_shape: tuple[int, ...]

    _lightning_module: Annotated[
        NeuralODEModule | None,
        PrivateAttr()
    ] = None

    def _to_mlflow(self, prefix=''):
        super()._to_mlflow(prefix=prefix)
        if prefix:
            prefix += '.'
        self.vector_field._to_mlflow(prefix=prefix + 'vector_field')
        mlflow.log_param(
            f"{prefix}base_distribution",
            self.base_distribution
        )
        mlflow.log_param(
            f"{prefix}integration_times",
            json.dumps(self.integration_times)
        )
        mlflow.log_param(
            f"{prefix}learning_rate",
            self.learning_rate
        )
        mlflow.log_param(
            f"{prefix}adjoint",
            self.adjoint
        )
        mlflow.log_param(
            f"{prefix}solver",
            self.solver
        )
        mlflow.log_param(
            f"{prefix}atol",
            self.atol
        )
        mlflow.log_param(
            f"{prefix}rtol",
            self.rtol
        )
        mlflow.log_param(
            f"{prefix}input_shape",
            json.dumps(self.input_shape)
        )

    @classmethod
    def from_mlflow(cls,
                    mlflow_run: Run,
                    prefix='', **kwargs):
        kwargs = super().from_mlflow(mlflow_run, prefix=prefix, **kwargs)
        if prefix:
            prefix += '.'
        model_type: Type[MLFlowLoggedModel] = cls.model_fields['vector_field'].annotation
        kwargs['vector_field'] = model_type.from_mlflow(
            mlflow_run,
            prefix=prefix + 'vector_field',
        )
        kwargs['base_distribution'] = mlflow_run.data.params.get(
            f'{prefix}base_distribution',
            cls.model_fields['base_distribution'].default
        )
        integration_times_str = mlflow_run.data.params.get(
            f'{prefix}integration_times',
            None
        )
        if integration_times_str is not None:
            kwargs['integration_times'] = json.loads(integration_times_str)
        else:
            kwargs['integration_times'] = cls.model_fields[
                'integration_times'].default
        kwargs['learning_rate'] = float(
            mlflow_run.data.params.get(
                f'{prefix}learning_rate',
                cls.model_fields['learning_rate'].default
            )
        )
        kwargs['adjoint'] = mlflow_run.data.params.get(
            f'{prefix}adjoint',
            str(cls.model_fields['adjoint'].default)
        ) == 'True'
        kwargs['solver'] = mlflow_run.data.params.get(
            f'{prefix}solver',
            cls.model_fields['solver'].default
        )
        kwargs['atol'] = float(
            mlflow_run.data.params.get(
                f'{prefix}atol',
                cls.model_fields['atol'].default
            )
        )
        kwargs['rtol'] = float(
            mlflow_run.data.params.get(
                f'{prefix}rtol',
                cls.model_fields['rtol'].default
            )
        )
        kwargs['input_shape'] = tuple(
            json.loads(
                mlflow_run.data.params[f'{prefix}input_shape']
            )
        )
        instance = cls(**kwargs)
        return instance

    def sample(self, shape: tuple[int],
               context: torch.Tensor | None = None):
        return self.lightning_module.sample(shape)

    def trajectory(self,
                   x: torch.Tensor,
                   integration_times: torch.Tensor) -> torch.Tensor:
        return self.lightning_module.trajectory(
            x=x,
            integration_times=integration_times
        )


class TimeEmbeddingMLPNeuralODEModel(NeuralODEModel):
    vector_field: TimeEmbeddingMLPConfig
