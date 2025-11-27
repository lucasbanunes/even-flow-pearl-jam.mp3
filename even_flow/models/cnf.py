from functools import partial
from abc import abstractmethod
from typing import Annotated, ClassVar, Literal, Any
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from datetime import datetime, timezone
from pydantic import Field, PrivateAttr
import torch
from torch.func import vmap
from torchdiffeq import odeint, odeint_adjoint
from torchmetrics import MeanMetric, MetricCollection
import mlflow
import lightning as L
import json
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .mlp import TimeEmbeddingMLPConfig
from ..mlflow import tmp_artifact_download, MLFlowLoggedClass
from ..torch import memory_optimized_divergence1d
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
    Literal['memory_optimized', 'speed_optimized'],
    Field(
        help="Strategy for computing the divergence."
    )
]
type BaseDistributionType = Annotated[
    Literal['standard_normal', 'uniform'],
    Field(
        help="The base distribution for the CNF."
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
type DivergenceStrategyType = Annotated[
    DivergenceStrategyType,
    Field('memory_optimized', description="Strategy for computing the divergence")
]
type LearningRateType = Annotated[
    float,
    Field(1e-3, description="Learning rate for the optimizer")
]
type MetricModeType = Annotated[
    Literal['min', 'max'],
    Field('min', description="Mode for metric monitoring")
]


class CNF1D(L.LightningModule):

    def __init__(self,
                 model_config: 'CNF1DModel'):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.vector_field = model_config.vector_field.as_nn_module()
        self.adjoint = model_config.adjoint
        if self.adjoint:
            self.odeint_func = odeint_adjoint
        else:
            self.odeint_func = odeint

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
        if self.divergence_strategy == 'memory_optimized':
            self.divergence_strategy = memory_optimized_divergence1d
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
        div0 = torch.zeros_like(z0)
        z, int_div = self.odeint_func(
            self.augmented_function,
            (z0, div0),
            self.integration_times.type_as(z0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return z[-1], int_div[-1]

    def augmented_function(self,
                           t: torch.Tensor,
                           state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
        fixed_t = partial(self.vector_field, t)
        divergence_func = self.divergence_strategy(fixed_t)
        vectorized_dvergence_func = vmap(divergence_func)
        z_eval, divergence = vectorized_dvergence_func(z)
        return z_eval, divergence

    def log_prob(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zf, div_int = self.forward(z0)
        logp_zf = self.base_distribution.log_prob(zf)
        return zf, div_int, logp_zf + div_int

    def sample(self, num_samples: int) -> torch.Tensor:
        zf = self.base_distribution.sample((num_samples,))
        integration_times_reversed = torch.flip(self.integration_times, [0])
        z, _ = self.odeint_func(
            self.vector_field,
            zf,
            integration_times_reversed.type_as(zf),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return z[-1]

    def training_step(self,
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        _, _, log_prob = self.log_prob(batch)
        loss = -log_prob.mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        x, y = batch
        y_pred = self(x)
        self.val_metrics.update(y_pred, y)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor):
        x, y = batch
        y_pred = self(x)
        self.test_metrics.update(y_pred, y)

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer


class CNF1DModel(MLFlowLoggedModel):

    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'cnf1d.ckpt'

    vector_field: Annotated[
        MLFlowLoggedClass,
        Field(
            description="The vector field defining the CNF transformation."
        )
    ]
    adjoint: AdjointType = True
    base_distribution: BaseDistributionType = 'standard_normal'
    integration_times: IntegrationTimesType = [0.0, 1.0]
    solver: SolverType = 'dopri5'
    atol: AtolType = 1e-5
    rtol: RtolType = 1e-5
    divergence_strategy: DivergenceStrategyType = 'memory_optimized'
    learning_rate: LearningRateType = 1e-3
    accelerator: TrainerAcceleratorType = 'cpu'
    checkpoints_dir: CheckpointsDirType = None
    max_epochs: MaxEpochsType = 3
    patience: PatienceType = None
    verbose: TrainerTestVerbosityType = True
    mode: MetricModeType

    _lightning_module: Annotated[
        CNF1D | None,
        PrivateAttr()
    ] = None

    @property
    def lightning_module(self) -> CNF1D:
        return self._lightning_module

    # _trainer: Annotated[
    #     L.Trainer | None,
    #     PrivateAttr()
    # ] = None

    # @property
    # def trainer(self) -> L.Trainer:
    #     return self._trainer

    def model_post_init(self, context):
        if self._lightning_module is None:
            self._lightning_module = CNF1D(model_config=self)

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
        mlflow.log_param(f'{prefix}patience', str(self.patience))
        mlflow.log_param(f'{prefix}verbose', str(self.verbose))

    @classmethod
    @abstractmethod
    def from_mlflow(cls, mlflow_run, prefix=''):
        raise NotImplementedError(
            "from_mlflow method must be implemented by subclasses.")

    def get_mlflow_logger(self) -> MLFlowLogger:
        mlflow_client = mlflow.MlflowClient()
        run = mlflow.active_run()
        experiment = mlflow_client.get_experiment(run.info.experiment_id)
        lightning_mlflow_logger = MLFlowLogger(
            run_name=self.name,
            run_id=self.id_,
            tracking_uri=mlflow.mlflow.get_tracking_uri(),
            experiment_name=experiment.name
        )
        return lightning_mlflow_logger

    def get_lightning_module_artifact_name(self, prefix='') -> str:
        if prefix:
            prefix += prefix.replace('.', '_') + '_'
        artifact_name = f'{prefix}{self.LIGHTNING_MODULE_ARTIFACT_PATH}'
        return artifact_name

    def fit(self,
            datamodule: L.LightningDataModule,
            prefix: str = '') -> L.Trainer:
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
            self.lightning_module = CNF1D.load_from_checkpoint(
                checkpoint.best_model_path)
            checkpoint_temp_path = Path(
                tmp_dir) / self.get_lightning_module_artifact_name(prefix)
            shutil.copy(checkpoint.best_model_path, checkpoint_temp_path)
            mlflow.log_artifact(str(checkpoint_temp_path))
            mlflow.pytorch.log_model(
                pytorch_model=self.lightning_module,
                artifact_path='model')
            shutil.rmtree(str(checkpoints_dir))

        return trainer

    def evaluate(self,
                 dataloader: torch.utils.data.DataLoader) -> dict[str, Any]:
        trainer = L.Trainer(
            accelerator=self.accelerator,
            devices=1,
            enable_progress_bar=self.verbose,
        )
        trainer.test(self.lightning_module, dataloaders=dataloader,
                     verbose=self.verbose)
        metrics = self.lightning_module.get_test_metrics()
        self.lightning_module.reset_metrics()
        return metrics


class TimeEmbeddingMLPCNF1DModel(CNF1DModel):
    vector_field: TimeEmbeddingMLPConfig

    @classmethod
    def from_mlflow(cls, mlflow_run, prefix=''):
        kwargs = {}
        kwargs['adjoint'] = bool(mlflow_run.data.params.get(f'{prefix}adjoint',
                                                            cls.model_fields['adjoint'].default))
        kwargs['base_distribution'] = mlflow_run.data.params.get(f'{prefix}base_distribution',
                                                                 cls.model_fields['base_distribution'].default)
        kwargs['integration_times'] = mlflow_run.data.params.get(f'{prefix}integration_times',
                                                                 cls.model_fields['integration_times'].default)
        if kwargs['integration_times'] is not None:
            kwargs['integration_times'] = json.loads(
                kwargs['integration_times'])
        kwargs['solver'] = mlflow_run.data.params.get(f'{prefix}solver',
                                                      cls.model_fields['solver'].default)
        kwargs['atol'] = float(mlflow_run.data.params.get(f'{prefix}atol',
                                                          cls.model_fields['atol'].default))
        kwargs['rtol'] = float(mlflow_run.data.params.get(f'{prefix}rtol',
                                                          cls.model_fields['rtol'].default))
        kwargs['divergence_strategy'] = mlflow_run.data.params.get(f'{prefix}divergence_strategy',
                                                                   cls.model_fields['divergence_strategy'].default)
        kwargs['learning_rate'] = float(mlflow_run.data.params.get(f'{prefix}learning_rate',
                                                                   cls.model_fields['learning_rate'].default))
        kwargs['accelerator'] = mlflow_run.data.params.get(f'{prefix}accelerator',
                                                           cls.model_fields['accelerator'].default)
        kwargs['checkpoints_dir'] = mlflow_run.data.params.get(f'{prefix}checkpoints_dir',
                                                               cls.model_fields['checkpoints_dir'].default)
        kwargs['max_epochs'] = int(mlflow_run.data.params.get(f'{prefix}max_epochs',
                                                              cls.model_fields['max_epochs'].default))
        patience_param = mlflow_run.data.params.get(f'{prefix}patience',
                                                    cls.model_fields['patience'].default)
        if patience_param == 'None':
            kwargs['patience'] = None
        else:
            kwargs['patience'] = int(patience_param)
        kwargs['verbose'] = bool(mlflow_run.data.params.get(f'{prefix}verbose',
                                                            cls.model_fields['verbose'].default))
        kwargs['vector_field'] = TimeEmbeddingMLPConfig.from_mlflow(
            mlflow_run,
            prefix=f'{prefix}vector_field'
        )
        instance = cls(**kwargs)
        with tmp_artifact_download(
            run_id=mlflow_run.info.run_id,
            artifact_path=instance.get_lightning_module_artifact_name(prefix)
        ) as ckpt_path:
            instance._lightning_module = CNF1D.load_from_checkpoint(ckpt_path)
        return instance
