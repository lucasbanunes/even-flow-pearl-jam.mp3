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

from .dataset import MoonsDataset
from .models import TimeEmbeddingMLPNeuralODEClassifier
from ..jobs import MLFlowBaseModel, DEFAULT_TRAINING_JOB_METRICS
from ..utils import get_logger
from ..pydantic import YamlBaseModel


class MoonsTimeEmbeddinngMLPNeuralODE(MLFlowBaseModel, YamlBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_CKPT_PATH: ClassVar[str] = 'model.ckpt'
    MLFLOW_LOGGER_ATTRIBUTES: ClassVar[list[str] | None] = None

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
    def from_mlflow(cls, mlflow_run) -> Self:
        return cls(
            model=TimeEmbeddingMLPNeuralODEClassifier.from_mlflow(
                mlflow_run,
                model_name="model"
            )
        )

    def to_mlflow(self, prefix: str = ""):
        raise NotImplementedError()

    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):
        logger = get_logger()
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
        self.model.to_mlflow()
        logger.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(checkpoint_dir))
