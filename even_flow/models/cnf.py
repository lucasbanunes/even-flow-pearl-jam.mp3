from typing import Annotated, ClassVar, Literal, Any, Type
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from datetime import datetime, timezone
from pydantic import Field, PrivateAttr
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from torchmetrics import MeanMetric, MetricCollection
import mlflow
import lightning as L
import json
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow.models.model import ModelInfo

from .mlp import TimeEmbeddingMLPConfig
from ..mlflow import tmp_artifact_download, MLFlowLoggedClass
from ..lightning import (
    CheckpointsDirType,
    MaxEpochsType,
    PatienceType,
    TrainerAcceleratorType,
    TrainerTestVerbosityType,
)
from ..pydantic import MLFlowLoggedModel
from ..utils import get_logger


type AdjointType = Annotated[
    bool,
    Field(True, description="Whether to use the adjoint method for backpropagation")
]
type DivergenceStrategyType = Annotated[
    Literal['exact', ],
    Field(
        description="Strategy for computing the divergence."
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
type SolverType = Annotated[
    str,
    Field('dopri5', description="The ODE solver to use")
]
type AtolType = Annotated[
    float,
    Field(1e-5, description="Absolute tolerance for the ODE solver")
]
type RtolType = Annotated[
    float,
    Field(1e-5, description="Relative tolerance for the ODE solver")
]
type LearningRateType = Annotated[
    float,
    Field(1e-3, description="Learning rate for the optimizer")
]
type MetricModeType = Annotated[
    Literal['min', 'max'],
    Field('min', description="Mode for metric monitoring")
]
type MonitorType = Annotated[
    str,
    Field(description="Metric to monitor during training")
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

        base_distribution = model_config.base_distribution
        if base_distribution is None or base_distribution == 'standard_normal':
            self.base_distribution = torch.distributions.Normal(
                loc=0.0, scale=1.0)
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

        self.divergence_strategy = model_config.divergence_strategy
        if self.divergence_strategy == 'exact':
            self.augmented_function = self.exact_divergence
        else:
            raise ValueError(
                f"Unsupported divergence strategy: {self.divergence_strategy}")

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

    def forward_no_adjoint(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        div0 = torch.zeros(z0.shape[0], 1, dtype=z0.dtype)

        z, int_div = odeint(
            self.augmented_function,
            (z0, div0),
            self.integration_times.type_as(z0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return z[-1], int_div[-1]

    def exact_divergence(self,
                         t: torch.Tensor,
                         state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
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

        # fixed_t = partial(self.vector_field, t)
        # divergence_func = self.divergence_strategy(fixed_t)
        # vectorized_dvergence_func = vmap(divergence_func)
        # z_eval, divergence = vectorized_dvergence_func(
        #     z.reshape(z.shape[0], 1, z.shape[1]))
        # z_eval = z_eval.squeeze()
        # divergence = divergence.reshape(divergence.shape[0], 1)

        return z, divergence

    def log_prob(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zf, div_int = self.forward(z0)
        logp_zf = self.base_distribution.log_prob(zf)
        return zf, div_int, logp_zf - div_int

    def sample(self, num_samples: int) -> torch.Tensor:
        zf = self.base_distribution.sample((num_samples,))
        integration_times_reversed = torch.flip(self.integration_times, [0])
        z, _ = odeint(
            self.vector_field,
            zf,
            integration_times_reversed.type_as(zf),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return z[-1]

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

    @torch.enable_grad()
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

    @torch.enable_grad()
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



class CNFModel(MLFlowLoggedModel):

    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'CNF.ckpt'

    vector_field: VectorFieldType
    adjoint: AdjointType = True
    base_distribution: BaseDistributionType = 'standard_normal'
    integration_times: IntegrationTimesType = [0.0, 1.0]
    solver: SolverType = 'dopri5'
    atol: AtolType = 1e-5
    rtol: RtolType = 1e-5
    divergence_strategy: DivergenceStrategyType = 'exact'
    learning_rate: LearningRateType = 1e-3
    accelerator: TrainerAcceleratorType = 'cpu'
    checkpoints_dir: CheckpointsDirType = None
    max_epochs: MaxEpochsType = 3
    patience: PatienceType = 3
    verbose: TrainerTestVerbosityType = True
    mode: MetricModeType = 'min'
    monitor: MonitorType = 'val_loss'
    num_sanity_val_steps: int = 5

    _lightning_module: Annotated[
        CNF | None,
        PrivateAttr()
    ] = None

    @property
    def lightning_module(self) -> CNF:
        return self._lightning_module

    def model_post_init(self, context):
        if self._lightning_module is None:
            self._lightning_module = CNF(model_config=self)

    def _to_mlflow(self, prefix=''):
        if prefix:
            prefix += '.'
        self.vector_field.to_mlflow(prefix=f'{prefix}vector_field')
        mlflow.log_param(f'{prefix}adjoint', self.adjoint)
        mlflow.log_param(f'{prefix}base_distribution', self.base_distribution)
        if self.integration_times == 'None':
            mlflow.log_param(f'{prefix}integration_times', None)
        else:
            mlflow.log_param(f'{prefix}integration_times',
                             json.dumps(self.integration_times))
        mlflow.log_param(f'{prefix}solver', self.solver)
        mlflow.log_param(f'{prefix}atol', self.atol)
        mlflow.log_param(f'{prefix}rtol', self.rtol)
        mlflow.log_param(f'{prefix}divergence_strategy',
                         self.divergence_strategy)
        mlflow.log_param(f'{prefix}learning_rate', self.learning_rate)
        mlflow.log_param(f'{prefix}accelerator', self.accelerator)
        mlflow.log_param(f'{prefix}checkpoints_dir', str(self.checkpoints_dir))
        mlflow.log_param(f'{prefix}max_epochs', self.max_epochs)
        mlflow.log_param(f'{prefix}patience', self.patience)
        mlflow.log_param(f'{prefix}verbose', self.verbose)
        mlflow.log_param(f'{prefix}mode', self.mode)
        mlflow.log_param(f'{prefix}monitor', self.monitor)
        mlflow.log_param(f'{prefix}num_sanity_val_steps',
                         self.num_sanity_val_steps)

    @classmethod
    def from_mlflow(cls, mlflow_run, prefix='') -> dict[str, Any]:
        formated_prefix = prefix
        if prefix:
            formated_prefix += '.'
        kwargs = {}
        kwargs['adjoint'] = bool(mlflow_run.data.params.get(f'{formated_prefix}adjoint',
                                                            cls.model_fields['adjoint'].default))
        kwargs['base_distribution'] = mlflow_run.data.params.get(f'{prefix}base_distribution',
                                                                 cls.model_fields['base_distribution'].default)
        kwargs['integration_times'] = mlflow_run.data.params.get(f'{formated_prefix}integration_times',
                                                                 cls.model_fields['integration_times'].default)
        if isinstance(kwargs['integration_times'], str):
            kwargs['integration_times'] = json.loads(
                kwargs['integration_times'])
        kwargs['solver'] = mlflow_run.data.params.get(f'{formated_prefix}solver',
                                                      cls.model_fields['solver'].default)
        kwargs['atol'] = float(mlflow_run.data.params.get(f'{formated_prefix}atol',
                                                          cls.model_fields['atol'].default))
        kwargs['rtol'] = float(mlflow_run.data.params.get(f'{formated_prefix}rtol',
                                                          cls.model_fields['rtol'].default))
        kwargs['divergence_strategy'] = mlflow_run.data.params.get(f'{formated_prefix}divergence_strategy',
                                                                   cls.model_fields['divergence_strategy'].default)
        kwargs['learning_rate'] = float(mlflow_run.data.params.get(f'{formated_prefix}learning_rate',
                                                                   cls.model_fields['learning_rate'].default))
        kwargs['accelerator'] = mlflow_run.data.params.get(f'{formated_prefix}accelerator',
                                                           cls.model_fields['accelerator'].default)
        kwargs['checkpoints_dir'] = mlflow_run.data.params.get(f'{formated_prefix}checkpoints_dir',
                                                               cls.model_fields['checkpoints_dir'].default)
        if kwargs['checkpoints_dir'] == 'None':
            kwargs['checkpoints_dir'] = None
        kwargs['max_epochs'] = int(mlflow_run.data.params.get(f'{formated_prefix}max_epochs',
                                                              cls.model_fields['max_epochs'].default))
        kwargs['patience'] = mlflow_run.data.params.get(f'{formated_prefix}patience',
                                                        cls.model_fields['patience'].default)
        kwargs['patience'] = int(kwargs['patience'])
        kwargs['verbose'] = bool(mlflow_run.data.params.get(f'{formated_prefix}verbose',
                                                            cls.model_fields['verbose'].default))
        kwargs['mode'] = mlflow_run.data.params.get(f'{formated_prefix}mode',
                                                    cls.model_fields['mode'].default)
        kwargs['monitor'] = mlflow_run.data.params.get(f'{formated_prefix}monitor',
                                                       cls.model_fields['monitor'].default)
        kwargs['num_sanity_val_steps'] = int(mlflow_run.data.params.get(f'{formated_prefix}num_sanity_val_steps',
                                                                        cls.model_fields['num_sanity_val_steps'].default))
        vector_field_type: Type[MLFlowLoggedClass] = cls.model_fields['vector_field'].annotation
        kwargs['vector_field'] = vector_field_type.from_mlflow(
            mlflow_run,
            prefix=f'{formated_prefix}vector_field'
        )
        instance = cls(**kwargs)
        with tmp_artifact_download(
            run_id=mlflow_run.info.run_id,
            artifact_path=instance.get_lightning_module_artifact_name(prefix)
        ) as ckpt_path:
            instance._lightning_module = CNF.load_from_checkpoint(ckpt_path)
        return instance

    def get_mlflow_logger(self) -> MLFlowLogger:
        mlflow_client = mlflow.MlflowClient()
        active_run = mlflow.active_run()
        experiment = mlflow_client.get_experiment(
            active_run.info.experiment_id)
        lightning_mlflow_logger = MLFlowLogger(
            run_name=active_run.data.tags['mlflow.runName'],
            run_id=active_run.info.run_id,
            tracking_uri=mlflow.mlflow.get_tracking_uri(),
            experiment_name=experiment.name
        )
        return lightning_mlflow_logger

    def get_lightning_module_artifact_name(self, prefix='') -> str:
        if prefix:
            prefix = prefix.replace('.', '_') + '_'
        artifact_name = f'{prefix}{self.LIGHTNING_MODULE_ARTIFACT_PATH}'
        return artifact_name

    def fit(self,
            datamodule: L.LightningDataModule,
            prefix: str = '') -> tuple[L.Trainer, ModelInfo]:
        logger = get_logger()
        logger.debug('Creating trainer...')
        lightning_mlflow_logger = self.get_mlflow_logger()

        callbacks = []
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            if self.checkpoints_dir is None:
                checkpoints_dir = tmp_dir / "checkpoints"
            else:
                checkpoints_dir = self.checkpoints_dir
            checkpoint = ModelCheckpoint(
                monitor=self.monitor,  # Monitor a validation metric
                dirpath=checkpoints_dir,  # Directory to save checkpoints
                filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
                save_top_k=3,
                mode=self.mode,  # Save based on maximum validation accuracy
                save_on_train_epoch_end=False
            )

            callbacks = [
                EarlyStopping(
                    monitor=self.monitor,
                    patience=self.patience,
                    mode=self.mode,
                    min_delta=1e-3
                ),
                checkpoint,
            ]
            trainer = L.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=1,
                logger=lightning_mlflow_logger,
                callbacks=callbacks,
                enable_progress_bar=self.verbose,
                enable_model_summary=self.verbose,
                num_sanity_val_steps=self.num_sanity_val_steps
            )

            logger.debug('Starting training process...')
            fit_start = datetime.now(timezone.utc)
            mlflow.log_metric('fit_start', fit_start.timestamp())
            trainer.fit(self.lightning_module, datamodule=datamodule)
            fit_end = datetime.now(timezone.utc)
            mlflow.log_metric('fit_end', fit_end.timestamp())
            mlflow.log_metric(
                "fit_duration", (fit_end - fit_start).total_seconds())
            logger.debug('Logging model artifacts to MLflow...')
            self._lightning_module = CNF.load_from_checkpoint(
                checkpoint.best_model_path)
            checkpoint_temp_path = Path(
                tmp_dir) / self.get_lightning_module_artifact_name(prefix)
            shutil.copy(checkpoint.best_model_path, checkpoint_temp_path)
            mlflow.log_artifact(str(checkpoint_temp_path))
            model_info = mlflow.pytorch.log_model(
                pytorch_model=self.lightning_module,
                artifact_path='model')
            shutil.rmtree(str(checkpoints_dir))

        return trainer, model_info

    def evaluate(self,
                 dataloader: torch.utils.data.DataLoader) -> dict[str, Any]:
        # Using trainer.test doesn´t allow computing the internal
        # jacobian necessary to compute the log_prob, so we manually
        # run the test loop here.
        for i, batch in enumerate(dataloader):
            self.lightning_module.test_step(batch, i)
        metrics = self.lightning_module.get_test_metrics()
        self.lightning_module.reset_metrics()
        return metrics


class TimeEmbeddingMLPCNFModel(CNFModel):
    vector_field: TimeEmbeddingMLPConfig
