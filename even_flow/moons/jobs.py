from pathlib import Path
from typing import Annotated, Self, ClassVar
import shutil
from datetime import datetime, timezone
import mlflow.entities
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow
from mlflow.entities import Run

from .dataset import MoonsDataset
from .models import TimeEmbeddingMLPNeuralODEClassifier
from ..jobs import MLFlowBaseModel, DEFAULT_TRAINING_JOB_METRICS
from ..utils import get_logger
from ..pydantic import YamlBaseModel


class MoonsTimeEmbeddinngMLPNeuralODEJob(MLFlowBaseModel, YamlBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_CKPT_PATH: ClassVar[str] = 'model.ckpt'
    MLFLOW_LOGGER_ATTRIBUTES: ClassVar[list[str] | None] = None
    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    MODEL_PREFIX: ClassVar[str] = 'model'

    max_epochs: int
    model: Annotated[
        TimeEmbeddingMLPNeuralODEClassifier,
        BeforeValidator(
            TimeEmbeddingMLPNeuralODEClassifier.pydantic_before_validator),
        PlainSerializer(
            TimeEmbeddingMLPNeuralODEClassifier.pydantic_plain_serializer)
    ]
    monitor: str

    accelerator: str = 'cpu'
    checkpoints_dir: Path | None = None
    datamodule: Annotated[
        MoonsDataset,
        BeforeValidator(MoonsDataset.pydantic_before_validator),
        PlainSerializer(MoonsDataset.pydantic_plain_serializer)
    ] = MoonsDataset()
    metrics: dict[str, dict[str, float]] = DEFAULT_TRAINING_JOB_METRICS.copy()
    patience: int = 10

    @classmethod
    def from_mlflow(cls, mlflow_run: Run, prefix: str = '') -> Self:
        if prefix:
            prefix += "."
        return cls(
            max_epochs=int(mlflow_run.data.params[f'{prefix}max_epochs']),
            model=TimeEmbeddingMLPNeuralODEClassifier.from_mlflow(
                mlflow_run,
                prefix=f'{prefix}{cls.MODEL_PREFIX}'
            ),
            monitor=mlflow_run.data.params[f'{prefix}monitor'],
            accelerator=mlflow_run.data.params[f'{prefix}accelerator'],
            checkpoints_dir=mlflow_run.data.params.get(f'{prefix}checkpoints_dir', None),
            datamodule=MoonsDataset.from_mlflow(mlflow_run,
                                                prefix=f'{prefix}{cls.DATAMODULE_PREFIX}'),
            patience=int(mlflow_run.data.params[f'{prefix}patience']),
        )

    def to_mlflow(self, prefix: str = '') -> None:
        if prefix:
            prefix += "."
        mlflow.log_param(f'{prefix}max_epochs', self.max_epochs)
        self.model.to_mlflow(prefix=f'{prefix}{self.MODEL_PREFIX}')
        mlflow.log_param(f'{prefix}monitor', self.monitor)
        mlflow.log_param(f'{prefix}accelerator', self.accelerator)
        if self.checkpoints_dir:
            mlflow.log_param(f'{prefix}checkpoints_dir', str(self.checkpoints_dir))
        self.datamodule.to_mlflow(prefix=f'{prefix}{self.DATAMODULE_PREFIX}')
        mlflow.log_param(f'{prefix}patience', self.patience)

    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):

        logger = get_logger()

        mlflow.log_param('max_epochs', self.max_epochs)
        mlflow.log_param('monitor', self.monitor)
        mlflow.log_param('accelerator', self.accelerator)
        if self.checkpoints_dir:
            mlflow.log_param('checkpoints_dir', str(self.checkpoints_dir))
        mlflow.log_param('patience', self.patience)
        self.datamodule.to_mlflow(prefix=self.DATAMODULE_PREFIX)

        mlflow_client = mlflow.MlflowClient()
        experiment = mlflow_client.get_experiment(run.info.experiment_id)
        lightning_mlflow_logger = MLFlowLogger(
            run_name=self.name,
            run_id=self.id_,
            tracking_uri=mlflow.mlflow.get_tracking_uri(),
            experiment_name=experiment.name
        )

        if self.checkpoints_dir is None:
            checkpoint_dir = tmp_dir / "checkpoints"
        else:
            checkpoint_dir = self.checkpoints_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            monitor=self.monitor,  # Monitor a validation metric
            dirpath=checkpoint_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="max",  # Save based on maximum validation accuracy
            save_on_train_epoch_end=False
        )

        callbacks = [
            EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                mode="max",
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
            enable_progress_bar=False,
        )
        logger.info('Starting training process...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        trainer.fit(self.model, datamodule=self.datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        self.model = TimeEmbeddingMLPNeuralODEClassifier.load_from_checkpoint(
            checkpoint.best_model_path)
        self.model.to_mlflow(prefix='model',
                             checkpoint_path=checkpoint.best_model_path)
        logger.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(checkpoint_dir))
