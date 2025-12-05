from typing import Annotated, ClassVar, Literal, Any, Type
from pydantic import Field
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from torchmetrics import MeanMetric, MetricCollection
import mlflow
from mlflow.entities import Run
import lightning as L
import json
from zuko.flows.continuous import CNF as ZukoCNF

from .mlp import TimeEmbeddingMLPConfig
from .real_nvp import ActivationType, HiddenFeaturesType
from .torch import BaseNNModule, TorchModel
from .lightning import (
    LearningRateType,
    LightningModel,
)
from ..torch import TORCH_MODULES
from ..mlflow import MLFlowLoggedClass


type AdjointType = Annotated[
    bool,
    Field(description="Whether to use the adjoint method for backpropagation")
]
type BaseDistributionType = Annotated[
    Literal['standard_normal', 'uniform'],
    Field(
        description="The base distribution for the CNF."
    )
]
type IntegrationTimesType = Annotated[
    list[float] | None,
    Field(description="The timespan for integration")
]
type SolverType = Annotated[
    str,
    Field(description="The ODE solver to use")
]
type AtolType = Annotated[
    float,
    Field(description="Absolute tolerance for the ODE solver")
]
type RtolType = Annotated[
    float,
    Field(description="Relative tolerance for the ODE solver")
]

type VectorFieldType = Annotated[
    MLFlowLoggedClass,
    Field(description="The vector field defining the CNF transformation.")
]


class CNF(L.LightningModule):

    def __init__(self,
                 model_config: 'CNFModel'):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.vector_field: nn.Module = model_config.vector_field.as_nn_module()
        self.adjoint = model_config.adjoint
        if not self.adjoint:
            raise ValueError("Only adjoint method is currently supported.")

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

        integration_times = model_config.integration_times
        if integration_times is None:
            self.integration_times = torch.Tensor(
                [0.0, 1.0]).float()
        else:
            self.integration_times = torch.Tensor(
                integration_times).float()

        self.solver = model_config.solver
        self.atol = model_config.atol
        self.rtol = model_config.rtol
        self.learning_rate = model_config.learning_rate

        self.train_metrics = MetricCollection({
            'loss': MeanMetric()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.vector_field.reset_metrics()

    def get_test_metrics(self) -> dict[str, Any]:
        metrics = self.test_metrics.compute()
        metrics['nfe'] = self.vector_field.nfe
        return metrics

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        div0 = self.compute_divergence(self.integration_times[0],
                                       z0)
        # div0 = torch.zeros(z0.shape[0], 1, dtype=z0.dtype)

        z, int_div = odeint_adjoint(
            self.augmented_function,
            (z0, div0),
            self.integration_times.type_as(z0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_params=self.vector_field.parameters()
        )
        return z[-1], int_div[-1]

    def augmented_function(self,
                           t: torch.Tensor,
                           state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
        divergence = self.compute_divergence(t, z)

        return z, divergence.reshape(-1, 1)

    def compute_divergence(self,
                           t: torch.Tensor,
                           z: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = z.clone().requires_grad_(True)
            dzdt = self.vector_field.forward(t, z)
            grad_outputs = torch.eye(
                z.shape[-1], dtype=z.dtype, device=z.device)
            grad_outputs = grad_outputs.expand(*z.shape, -1).movedim(-1, 0)
            (jacobian,) = torch.autograd.grad(
                dzdt, z,
                grad_outputs=grad_outputs,
                create_graph=True, is_grads_batched=True
            )
            divergence = torch.einsum("i...i", jacobian)

        return divergence.reshape(-1, 1)

    def log_prob(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zf, div_int = self.forward(z0)
        logp_zf = self.base_distribution.log_prob(zf).reshape(-1, 1)
        return zf, div_int, logp_zf - div_int

    def training_step(self,
                      batch: tuple[torch.Tensor],
                      batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: tuple[torch.Tensor],
                        batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        self.val_metrics.update(loss)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self,
                  batch: tuple[torch.Tensor],
                  batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        self.test_metrics.update(loss)

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def sample(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            zf = self.base_distribution.sample((num_samples,))
            integration_times_reversed = torch.flip(self.integration_times,
                                                    dims=[0])
            z0 = odeint(
                self.vector_field,
                zf,
                integration_times_reversed.type_as(zf),
                method=self.solver,
                atol=self.atol,
                rtol=self.rtol
            )
            return zf, z0[-1]


class CNFHutchingson(CNF):

    def __init__(self,
                 model_config: 'CNFHutchingsonModel'):
        super().__init__(model_config=model_config)
        self.hutchingson_distribution = model_config.hutchingson_distribution
        match self.hutchingson_distribution:
            case 'standard_normal':
                self.noise_distribution = torch.distributions.Normal(
                    loc=0.0, scale=1.0)
            case 'rademacher':
                self.noise_distribution = torch.distributions.Bernoulli(
                    probs=0.5)
            case _:
                raise ValueError(
                    f"Unsupported Hutchingson distribution: {self.hutchingson_distribution}")

    @torch.enable_grad()
    def augmented_function(self,
                           t: torch.Tensor,
                           state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
        z = z.clone().requires_grad_(True)
        dzdt = self.vector_field.forward(t, z)
        grad_outputs = self.noise_distribution.sample(z.shape)
        (epsjp,) = torch.autograd.grad(
            dzdt, z, grad_outputs=grad_outputs, create_graph=True)
        divergence = (epsjp * grad_outputs).sum(dim=-1)

        return z, divergence.reshape(-1, 1)


class CNFModel(LightningModel):

    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'CNF.ckpt'
    LIGHTNING_MODULE_TYPE: ClassVar[Type[CNF]] = CNF

    vector_field: VectorFieldType
    adjoint: AdjointType = True
    base_distribution: BaseDistributionType = 'standard_normal'
    integration_times: IntegrationTimesType = [0.0, 1.0]
    solver: SolverType = 'dopri5'
    atol: AtolType = 1e-5
    rtol: RtolType = 1e-5
    learning_rate: LearningRateType = 1e-3
    input_shape: tuple[int, ...]

    lightning_module: Annotated[
        CNF | None,
        Field(description="The Lightning module instance.")
    ] = None

    def get_new_lightning_module(self):
        return CNF(self)

    def _to_mlflow(self, prefix=''):
        super()._to_mlflow(prefix=prefix)
        self.vector_field.to_mlflow(prefix=f'{prefix}.vector_field')
        mlflow.log_param(f'{prefix}.adjoint', self.adjoint)
        mlflow.log_param(f'{prefix}.base_distribution', self.base_distribution)
        mlflow.log_param(f'{prefix}.integration_times',
                         json.dumps(self.integration_times))
        mlflow.log_param(f'{prefix}.solver', self.solver)
        mlflow.log_param(f'{prefix}.atol', self.atol)
        mlflow.log_param(f'{prefix}.rtol', self.rtol)
        mlflow.log_param(f'{prefix}.learning_rate', self.learning_rate)
        mlflow.log_param(f'{prefix}.input_shape', json.dumps(self.input_shape))

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        kwargs = super()._from_mlflow(mlflow_run, prefix, **kwargs)
        kwargs['adjoint'] = bool(mlflow_run.data.params.get(f'{prefix}.adjoint',
                                                            cls.model_fields['adjoint'].default))
        kwargs['base_distribution'] = mlflow_run.data.params.get(f'{prefix}.base_distribution',
                                                                 cls.model_fields['base_distribution'].default)
        kwargs['integration_times'] = mlflow_run.data.params.get(f'{prefix}.integration_times',
                                                                 cls.model_fields['integration_times'].default)
        if isinstance(kwargs['integration_times'], str):
            kwargs['integration_times'] = json.loads(
                kwargs['integration_times'])
        kwargs['solver'] = mlflow_run.data.params.get(f'{prefix}.solver',
                                                      cls.model_fields['solver'].default)
        kwargs['atol'] = float(mlflow_run.data.params.get(f'{prefix}.atol',
                                                          cls.model_fields['atol'].default))
        kwargs['rtol'] = float(mlflow_run.data.params.get(f'{prefix}.rtol',
                                                          cls.model_fields['rtol'].default))
        kwargs['learning_rate'] = float(mlflow_run.data.params.get(f'{prefix}.learning_rate',
                                                                   cls.model_fields['learning_rate'].default))
        kwargs['input_shape'] = json.loads(
            mlflow_run.data.params[f'{prefix}.input_shape'])
        vector_field_type: Type[MLFlowLoggedClass] = cls.model_fields['vector_field'].annotation
        kwargs['vector_field'] = vector_field_type.from_mlflow(
            mlflow_run,
            prefix=f'{prefix}.vector_field'
        )
        return kwargs

    def sample(self, shape: tuple[int]) -> torch.Tensor:
        if self.lightning_module is None:
            self.lightning_module = self.get_new_lightning_module()
        return self.lightning_module.sample(num_samples=shape[0])


class TimeEmbeddingMLPCNFModel(CNFModel):
    vector_field: TimeEmbeddingMLPConfig


type HutchingsonDistributionType = Annotated[
    Literal['standard_normal', 'rademacher'],
    Field(
        description="The distribution used for Hutchingson estimator."
    )
]


class CNFHutchingsonModel(CNFModel):
    LIGHTNING_MODULE_TYPE: ClassVar[Type[CNF]] = CNFHutchingson
    hutchingson_distribution: HutchingsonDistributionType = 'standard_normal'

    def _to_mlflow(self, prefix=''):
        formated_prefix = prefix
        if prefix:
            formated_prefix += '.'
        super()._to_mlflow(prefix)
        mlflow.log_param(
            f'{formated_prefix}hutchingson_distribution', self.hutchingson_distribution)

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        kwargs = super()._from_mlflow(mlflow_run, prefix, **kwargs)
        kwargs['hutchingson_distribution'] = mlflow_run.data.params.get(f'{prefix}.hutchingson_distribution',
                                                                        cls.model_fields['hutchingson_distribution'].default)
        return kwargs


class TimeEmbeddingMLPCNFHutchinsonModel(CNFHutchingsonModel):
    vector_field: TimeEmbeddingMLPConfig


class ZukoCNFLightningModule(L.LightningModule):
    def __init__(self,
                 model_config: 'ZukoCNFModel'):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model: nn.Module = ZukoCNF(
            features=model_config.features,
            context=model_config.context,
            freqs=model_config.freqs,
            exact=model_config.exact,
            atol=model_config.atol,
            rtol=model_config.rtol,
            hidden_features=model_config.hidden_features,
            activation=TORCH_MODULES[model_config.activation],
        )
        self.learning_rate = model_config.learning_rate

        self.train_metrics = MetricCollection({
            'loss': MeanMetric()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(context).log_prob(x).reshape(-1, 1)

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

    def get_test_metrics(self) -> dict[str, float]:
        return self.test_metrics.compute()

    def get_val_metrics(self) -> dict[str, float]:
        return self.val_metrics.compute()

    def get_train_metrics(self) -> dict[str, float]:
        return self.train_metrics.compute()


class ZukoCNFModel(LightningModel):
    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'cnf.ckpt'
    LIGHTNING_MODULE_TYPE: ClassVar[Type[L.LightningModule]
                                    ] = ZukoCNFLightningModule

    features: int
    context: int = 0
    freqs: int = 3
    atol: AtolType = 1e-5
    rtol: RtolType = 1e-5
    exact: bool = True
    learning_rate: LearningRateType = 1e-3
    hidden_features: HiddenFeaturesType
    activation: ActivationType | None = None

    lightning_module: Annotated[
        ZukoCNFLightningModule | None,
        Field(description="The Lightning module instance.")
    ] = None

    def get_dist(self, context=None):
        return self.lightning_module.model()

    def get_new_lightning_module(self):
        return ZukoCNFLightningModule(self)

    def _to_mlflow(self, prefix=''):
        super()._to_mlflow(prefix=prefix)
        if prefix:
            prefix += '.'
        mlflow.log_param(f"{prefix}features", self.features)
        mlflow.log_param(f"{prefix}context", self.context)
        mlflow.log_param(f"{prefix}freqs", self.freqs)
        mlflow.log_param(f"{prefix}atol", self.atol)
        mlflow.log_param(f"{prefix}rtol", self.rtol)
        mlflow.log_param(f"{prefix}exact", self.exact)
        mlflow.log_param(f"{prefix}learning_rate", self.learning_rate)
        mlflow.log_param(f"{prefix}hidden_features",
                         json.dumps(self.hidden_features))
        mlflow.log_param(f"{prefix}activation", self.activation)

    @classmethod
    def _from_mlflow(cls,
                     mlflow_run: Run,
                     prefix='', **kwargs) -> dict[str, Any]:
        kwargs = super()._from_mlflow(mlflow_run, prefix=prefix, **kwargs)
        if prefix:
            prefix += '.'
        kwargs['features'] = int(
            mlflow_run.data.params[f'{prefix}features'])
        kwargs['context'] = int(
            mlflow_run.data.params.get(f'{prefix}context',
                                       cls.model_fields['context'].default))
        kwargs['freqs'] = int(
            mlflow_run.data.params.get(f'{prefix}freqs',
                                       cls.model_fields['freqs'].default))
        kwargs['atol'] = float(
            mlflow_run.data.params.get(f'{prefix}atol',
                                       cls.model_fields['atol'].default))
        kwargs['rtol'] = float(
            mlflow_run.data.params.get(f'{prefix}rtol',
                                       cls.model_fields['rtol'].default))
        kwargs['exact'] = mlflow_run.data.params.get(f'{prefix}exact',
                                                     cls.model_fields['exact'].default)
        kwargs['exact'] = kwargs['exact'] in ['True', 'true', True]
        kwargs['learning_rate'] = float(
            mlflow_run.data.params.get(f'{prefix}learning_rate',
                                       cls.model_fields['learning_rate'].default))
        hidden_features_str = mlflow_run.data.params[f'{prefix}hidden_features']
        kwargs['hidden_features'] = json.loads(hidden_features_str)
        activation_str = mlflow_run.data.params.get(
            f'{prefix}activation', cls.model_fields['activation'].default
        )
        kwargs['activation'] = activation_str if activation_str != 'None' else None
        return kwargs

    @torch.no_grad()
    def sample(self, shape: tuple[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lightning_module is None:
            self.lightning_module = self.get_new_lightning_module()
        normalizing_flow = self.lightning_module.model()
        if normalizing_flow.base.has_rsample:
            zf = normalizing_flow.base.rsample(shape)
        else:
            zf = normalizing_flow.base.sample(shape)
        z0 = normalizing_flow.transform.inv(zf)
        return zf, z0


class CNFTorchModule(BaseNNModule):

    def __init__(self,
                 model_config: 'CNFTorchModel'):
        super().__init__()

        self.vector_field: nn.Module = model_config.vector_field.as_nn_module()
        self.adjoint = model_config.adjoint
        if not self.adjoint:
            raise ValueError("Only adjoint method is currently supported.")

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

        integration_times = model_config.integration_times
        if integration_times is None:
            self.integration_times = torch.Tensor(
                [0.0, 1.0]).float()
        else:
            self.integration_times = torch.Tensor(
                integration_times).float()

        self.solver = model_config.solver
        self.atol = model_config.atol
        self.rtol = model_config.rtol
        self.learning_rate = model_config.learning_rate

        self.train_metrics = MetricCollection({
            'loss': MeanMetric()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.vector_field.reset_metrics()

    def get_test_metrics(self) -> dict[str, Any]:
        metrics = self.test_metrics.compute()
        metrics['nfe'] = self.vector_field.nfe
        return metrics

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # div0 = self.compute_divergence(self.integration_times[0],
        #                                z0)
        div0 = torch.zeros(z0.shape[0], 1, dtype=z0.dtype)

        z, int_div = odeint_adjoint(
            self.augmented_function,
            (z0, div0),
            self.integration_times.type_as(z0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_params=self.vector_field.parameters()
        )
        return z[-1], int_div[-1]

    def augmented_function(self,
                           t: torch.Tensor,
                           state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
        divergence = self.compute_divergence(t, z)

        return z, divergence

    @torch.enable_grad()
    def compute_divergence(self,
                           t: torch.Tensor,
                           z: torch.Tensor) -> torch.Tensor:
        z = z.clone().requires_grad_(True)
        dzdt = self.vector_field.forward(t, z)
        grad_outputs = torch.eye(
            z.shape[-1], dtype=z.dtype, device=z.device)
        grad_outputs = grad_outputs.expand(*z.shape, -1).movedim(-1, 0)
        (jacobian,) = torch.autograd.grad(
            dzdt, z,
            grad_outputs=grad_outputs,
            create_graph=True, is_grads_batched=True
        )
        divergence = torch.einsum("i...i", jacobian)

        return divergence.reshape(-1, 1)

    def log_prob(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zf, div_int = self.forward(z0)
        logp_zf = self.base_distribution.log_prob(zf).reshape(-1, 1)
        return zf, div_int, logp_zf + div_int

    def training_step(self,
                      batch: tuple[torch.Tensor],
                      batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        return loss

    def validation_step(self,
                        batch: tuple[torch.Tensor],
                        batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        self.val_metrics.update(loss)

    def test_step(self,
                  batch: tuple[torch.Tensor],
                  batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch[0])
        loss = -log_prob.mean()
        self.test_metrics.update(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    @torch.no_grad()
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        zf = self.base_distribution.sample((num_samples,))
        integration_times_reversed = torch.flip(self.integration_times,
                                                dims=[0])
        z0 = odeint(
            self.vector_field,
            zf,
            integration_times_reversed.type_as(zf),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return zf, z0[-1]


class CNFTorchModel(TorchModel):

    TORCH_MODEL_TYPE: ClassVar[Type[CNFTorchModule]] = CNFTorchModule
    TORCH_MODEL_ARTIFACT_PATH: ClassVar[str] = 'cnf_torch.pt'

    vector_field: VectorFieldType
    adjoint: AdjointType = True
    base_distribution: BaseDistributionType = 'standard_normal'
    integration_times: IntegrationTimesType = [0.0, 1.0]
    solver: SolverType = 'dopri5'
    atol: AtolType = 1e-5
    rtol: RtolType = 1e-5
    learning_rate: LearningRateType = 1e-3
    input_shape: tuple[int, ...]

    torch_module: Annotated[
        CNFTorchModule | None,
        Field(
            description="The underlying Torch module. If None, it will be created when accessed."
        )
    ] = None

    @torch.no_grad()
    def sample(self, shape: tuple[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.torch_module is None:
            self.torch_module = self.get_new_torch_module()
        base, transformed = self.torch_module.sample(shape[0])
        return base, transformed

    def get_new_torch_module(self) -> CNFTorchModule:
        return CNFTorchModule(model_config=self)

    def _to_mlflow(self, prefix=''):
        super()._to_mlflow(prefix=prefix)
        self.vector_field.to_mlflow(prefix=f'{prefix}.vector_field')
        mlflow.log_param(f'{prefix}.adjoint', self.adjoint)
        mlflow.log_param(f'{prefix}.base_distribution', self.base_distribution)
        mlflow.log_param(f'{prefix}.integration_times',
                         json.dumps(self.integration_times))
        mlflow.log_param(f'{prefix}.solver', self.solver)
        mlflow.log_param(f'{prefix}.atol', self.atol)
        mlflow.log_param(f'{prefix}.rtol', self.rtol)
        mlflow.log_param(f'{prefix}.learning_rate', self.learning_rate)
        mlflow.log_param(f"{prefix}.input_shape", json.dumps(self.input_shape))

    @classmethod
    def _from_mlflow(cls,
                     mlflow_run: Run,
                     prefix='', **kwargs) -> dict[str, Any]:
        kwargs = super()._from_mlflow(mlflow_run, prefix=prefix, **kwargs)
        kwargs['adjoint'] = bool(mlflow_run.data.params.get(f'{prefix}.adjoint',
                                                            cls.model_fields['adjoint'].default))
        kwargs['base_distribution'] = mlflow_run.data.params.get(f'{prefix}.base_distribution',
                                                                 cls.model_fields['base_distribution'].default)
        kwargs['integration_times'] = mlflow_run.data.params.get(f'{prefix}.integration_times',
                                                                 cls.model_fields['integration_times'].default)
        if isinstance(kwargs['integration_times'], str):
            kwargs['integration_times'] = json.loads(
                kwargs['integration_times'])
        kwargs['solver'] = mlflow_run.data.params.get(f'{prefix}.solver',
                                                      cls.model_fields['solver'].default)
        kwargs['atol'] = float(mlflow_run.data.params.get(f'{prefix}.atol',
                                                          cls.model_fields['atol'].default))
        kwargs['rtol'] = float(mlflow_run.data.params.get(f'{prefix}.rtol',
                                                          cls.model_fields['rtol'].default))
        kwargs['learning_rate'] = float(mlflow_run.data.params.get(f'{prefix}.learning_rate',
                                                                   cls.model_fields['learning_rate'].default))
        input_shape_str = mlflow_run.data.params[f'{prefix}.input_shape']
        kwargs['input_shape'] = json.loads(input_shape_str)
        vector_field_type: Type[MLFlowLoggedClass] = cls.model_fields['vector_field'].annotation
        kwargs['vector_field'] = vector_field_type.from_mlflow(
            mlflow_run,
            prefix=f'{prefix}.vector_field'
        )
        return kwargs


class TimeEmbeddingMLPCNFTorchModel(CNFTorchModel):
    vector_field: TimeEmbeddingMLPConfig
