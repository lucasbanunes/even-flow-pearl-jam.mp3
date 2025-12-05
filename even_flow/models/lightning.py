from typing import Annotated, Any, Literal, ClassVar, Self, Type
from pydantic import Field, PrivateAttr
from pathlib import Path
import shutil
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping as LightningEarlyStopping
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import mlflow
from mlflow.entities import Run
from mlflow.models.model import ModelInfo
from abc import abstractmethod

from ..utils import get_logger
from ..pydantic import MLFlowLoggedModel
from ..mlflow import tmp_artifact_download


type CheckpointsDirType = Annotated[
    Path | None,
    Field(
        description="Directory to save model checkpoints."
    )
]


type MaxEpochsType = Annotated[
    int,
    Field(
        description="Maximum number of epochs for training."
    )
]


type PatienceType = Annotated[
    int,
    Field(
        description="Number of epochs with no improvement after which training will be stopped."
    )
]

type TrainerAcceleratorType = Annotated[
    str,
    Field(
        description="The accelerator to use for training (e.g., 'cpu', 'gpu')."
    )
]

type TrainerTestVerbosityType = Annotated[
    bool,
    Field(
        description="Whether to enable verbose logging during testing."
    )
]

type LearningRateType = Annotated[
    float,
    Field(description="Learning rate for the optimizer")
]

type MetricModeType = Annotated[
    Literal['min', 'max'],
    Field(description="Mode for metric monitoring")
]

type MonitorType = Annotated[
    str,
    Field(description="Metric to monitor during training")
]

type SaveTopKType = Annotated[
    int,
    Field(
        description="Number of top model checkpoints to save based on the monitored metric."
    )
]

type ProfilerType = Annotated[
    Literal['simple', 'advanced'] | None,
    Field(description="The profiler to use during training")
]


class ModelCheckpointConfig(MLFlowLoggedModel):
    dirpath: CheckpointsDirType = None
    monitor: MonitorType = 'val_loss'
    mode: MetricModeType = 'min'
    save_top_k: SaveTopKType = 3
    filename: str | None = None

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.filename is None:
            self.filename = 'checkpoint-{epoch:02d}-{' + self.monitor + ':.2f}'

    def get(self, **kwargs) -> ModelCheckpoint:
        base_kwargs = {
            'dirpath': str(self.dirpath),
            'monitor': self.monitor,
            'mode': self.mode,
            'save_top_k': self.save_top_k,
            'filename': self.filename
        }
        base_kwargs.update(kwargs)
        return ModelCheckpoint(save_on_train_epoch_end=True, **base_kwargs)

    def _to_mlflow(self, prefix: str = '') -> None:
        if prefix:
            prefix += '.'
        mlflow.log_param(f"{prefix}dirpath", str(self.dirpath))
        mlflow.log_param(f"{prefix}monitor", self.monitor)
        mlflow.log_param(f"{prefix}mode", self.mode)
        mlflow.log_param(f"{prefix}save_top_k", self.save_top_k)
        mlflow.log_param(f"{prefix}filename", self.filename)

    @classmethod
    def _from_mlflow(cls, mlflow_run: Run,
                     prefix: str = '', **kwargs) -> dict[str, Any]:
        if prefix:
            prefix += '.'
        params = mlflow_run.data.params
        kwargs['dirpath'] = params.get(f"{prefix}dirpath",
                                       cls.model_fields['dirpath'].default)
        if kwargs['dirpath'] == 'None':
            kwargs['dirpath'] = None
        kwargs['monitor'] = params.get(f"{prefix}monitor",
                                       cls.model_fields['monitor'].default)
        kwargs['mode'] = params.get(f"{prefix}mode",
                                    cls.model_fields['mode'].default)
        kwargs['save_top_k'] = int(params.get(f"{prefix}save_top_k",
                                              cls.model_fields['save_top_k'].default))
        kwargs['filename'] = params.get(f"{prefix}filename",
                                        cls.model_fields['filename'].default)
        return kwargs


class EarlyStopping(MLFlowLoggedModel):
    monitor: MonitorType = 'val_loss'
    patience: PatienceType = 3
    min_delta: float = 1e-3
    mode: MetricModeType = 'min'
    stopping_threshold: float | None = None

    _current_patience: Annotated[
        int,
        PrivateAttr()
    ] = 0
    _best: Annotated[
        float | None,
        PrivateAttr()
    ] = None

    def model_post_init(self, context):
        super().model_post_init(context)

        if self.mode == 'min':
            self._best = float('inf')
        else:  # mode == 'max'
            self._best = float('-inf')

    def as_lightning(self) -> LightningEarlyStopping:
        return LightningEarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            min_delta=self.min_delta,
            mode=self.mode,
            stopping_threshold=self.stopping_threshold
        )

    def should_stop(self, current: float) -> bool:
        if self.mode == 'min':
            return self.should_stop_min(current)
        else:  # mode == 'max'
            return self.should_stop_max(current)

    def should_stop_min(self, current: float) -> bool:
        if current <= self.stopping_threshold:
            return True

        diff = self._best - current
        if diff >= self.min_delta:
            self._best = current
            self._current_patience = 0
            return False
        else:
            self._current_patience += 1
            if self._current_patience >= self.patience:
                return True
            else:
                return False

    def should_stop_max(self, current: float) -> bool:
        if current >= self.stopping_threshold:
            return True

        diff = current - self._best
        if diff >= self.min_delta:
            self._best = current
            self._current_patience = 0
            return False
        else:
            self._current_patience += 1
            if self._current_patience >= self.patience:
                return True
            else:
                return False

    def _to_mlflow(self, prefix: str = '') -> None:
        mlflow.log_param(f"{prefix}.monitor", self.monitor)
        mlflow.log_param(f"{prefix}.patience", self.patience)
        mlflow.log_param(f"{prefix}.min_delta", self.min_delta)
        mlflow.log_param(f"{prefix}.mode", self.mode)
        mlflow.log_param(f"{prefix}.stopping_threshold",
                         self.stopping_threshold)

    @classmethod
    def _from_mlflow(cls, mlflow_run: Run,
                     prefix: str = '', **kwargs) -> dict[str, Any]:
        kwargs = {}
        params = mlflow_run.data.params
        kwargs['monitor'] = params.get(f"{prefix}.monitor",
                                       cls.model_fields['monitor'].default)
        kwargs['patience'] = int(params.get(f"{prefix}.patience",
                                            cls.model_fields['patience'].default))
        kwargs['min_delta'] = float(params.get(f"{prefix}.min_delta",
                                               cls.model_fields['min_delta'].default))
        kwargs['mode'] = params.get(f"{prefix}.mode",
                                    cls.model_fields['mode'].default)
        kwargs['stopping_threshold'] = params.get(f"{prefix}.stopping_threshold",
                                                  cls.model_fields['stopping_threshold'].default)
        if kwargs['stopping_threshold'] == 'None':
            kwargs['stopping_threshold'] = None
        return kwargs


def get_lightning_module_artifact_name(suffix: str, prefix='') -> str:
    if prefix and prefix.endswith('.'):
        prefix = prefix.replace('.', '_')
    elif prefix:
        prefix = prefix.replace('.', '_') + '_'
    artifact_name = f'{prefix}{suffix}'
    return artifact_name


class LightningModel(MLFlowLoggedModel):

    BASE_PROFILER_FILENAME: ClassVar[str] = 'profiler'
    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'DEFINE-A-NICE-NAME.ckpt'
    LIGHTNING_MODULE_TYPE: ClassVar[Type[L.LightningModule]
                                    ] = L.LightningModule

    accelerator: TrainerAcceleratorType = 'cpu'
    profiler: ProfilerType = 'simple'
    max_epochs: MaxEpochsType = 3
    verbose: TrainerTestVerbosityType = True
    num_sanity_val_steps: int = 5

    checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    early_stopping: EarlyStopping = EarlyStopping()

    lightning_module: Annotated[
        L.LightningModule | None,
        Field(
            description="The Lightning module associated with the model."
        )
    ] = None

    @abstractmethod
    def get_new_lightning_module(self) -> L.LightningModule:
        raise NotImplementedError()

    def fit(self,
            datamodule: L.LightningDataModule,
            prefix: str = '') -> tuple[L.Trainer, ModelInfo]:
        logger = get_logger()
        logger.debug('Creating trainer...')
        if self.lightning_module is None:
            self.lightning_module = self.get_new_lightning_module()
        lightning_mlflow_logger = self.get_mlflow_logger()

        callbacks = []
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            if self.checkpoint.dirpath is None:
                checkpoints_dir = tmp_dir / "checkpoints"
            else:
                checkpoints_dir = self.checkpoint.dirpath
            checkpoint = self.checkpoint.get(
                dirpath=checkpoints_dir
            )
            callbacks = [
                self.early_stopping.as_lightning(),
                checkpoint
            ]
            fit_profiler_filename = f'fit-{self.BASE_PROFILER_FILENAME}.txt'
            profiler_dir = tmp_dir
            profiler = self.get_profiler(
                dirpath=str(profiler_dir),
                filename=self.BASE_PROFILER_FILENAME
            )
            trainer = L.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=1,
                logger=lightning_mlflow_logger,
                callbacks=callbacks,
                enable_progress_bar=self.verbose,
                enable_model_summary=self.verbose,
                num_sanity_val_steps=self.num_sanity_val_steps,
                profiler=profiler
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
            mlflow.log_artifact(str(profiler_dir / fit_profiler_filename))
            self._lightning_module = self.LIGHTNING_MODULE_TYPE.load_from_checkpoint(
                checkpoint.best_model_path)
            checkpoint_temp_path = Path(
                tmp_dir) / get_lightning_module_artifact_name(self.LIGHTNING_MODULE_ARTIFACT_PATH, prefix)
            shutil.copy(checkpoint.best_model_path, checkpoint_temp_path)
            mlflow.log_artifact(str(checkpoint_temp_path))
            model_info = mlflow.pytorch.log_model(
                pytorch_model=self.lightning_module,
                name=self.LIGHTNING_MODULE_ARTIFACT_PATH.replace('.ckpt', ''))
            shutil.rmtree(str(checkpoints_dir))

        return trainer, model_info

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

    def get_profiler(self, **kwargs) -> SimpleProfiler | AdvancedProfiler:
        if self.profiler == 'simple':
            return SimpleProfiler(**kwargs)
        elif self.profiler == 'advanced':
            return AdvancedProfiler(**kwargs)
        else:
            raise ValueError(f"Unsupported profiler: {self.profiler}")

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

    def _to_mlflow(self, prefix=''):
        if prefix:
            prefix += '.'
        mlflow.log_param(f"{prefix}accelerator", self.accelerator)
        mlflow.log_param(f"{prefix}profiler", self.profiler)
        mlflow.log_param(f"{prefix}max_epochs", self.max_epochs)
        mlflow.log_param(f"{prefix}verbose", self.verbose)
        mlflow.log_param(f"{prefix}num_sanity_val_steps",
                         self.num_sanity_val_steps)
        self.checkpoint.to_mlflow(prefix=prefix + 'checkpoint')
        self.early_stopping.to_mlflow(prefix=prefix + 'early_stopping')

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        if prefix:
            prefix += '.'
        kwargs['accelerator'] = mlflow_run.data.params.get(
            f'{prefix}accelerator',
            cls.model_fields['accelerator'].default)
        kwargs['profiler'] = mlflow_run.data.params.get(
            f'{prefix}profiler',
            cls.model_fields['profiler'].default)
        kwargs['max_epochs'] = int(mlflow_run.data.params.get(
            f'{prefix}max_epochs',
            cls.model_fields['max_epochs'].default))
        verbose_str = mlflow_run.data.params.get(
            f'{prefix}verbose',
            str(cls.model_fields['verbose'].default))
        kwargs['verbose'] = verbose_str.lower() == 'true'
        kwargs['num_sanity_val_steps'] = int(mlflow_run.data.params.get(
            f'{prefix}num_sanity_val_steps',
            cls.model_fields['num_sanity_val_steps'].default))
        kwargs['checkpoint'] = ModelCheckpointConfig.from_mlflow(
            mlflow_run,
            prefix=prefix + 'checkpoint')
        kwargs['early_stopping'] = EarlyStopping.from_mlflow(
            mlflow_run,
            prefix=prefix + 'early_stopping')
        kwargs['lightning_module'] = cls.load_lightning_module_from_checkpoint(
            run_id=mlflow_run.info.run_id,
            prefix=prefix
        )
        return kwargs

    @classmethod
    def load_lightning_module_from_checkpoint(cls, run_id: str, prefix: str = '') -> Self:
        with tmp_artifact_download(
            run_id=run_id,
            artifact_path=get_lightning_module_artifact_name(
                cls.LIGHTNING_MODULE_ARTIFACT_PATH, prefix)
        ) as ckpt_path:
            return cls.LIGHTNING_MODULE_TYPE.load_from_checkpoint(
                ckpt_path)
