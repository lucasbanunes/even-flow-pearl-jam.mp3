from pathlib import Path
from typing import Annotated, Self
import shutil
from datetime import datetime, timezone
import mlflow.entities
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
import typer
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow

from even_flow.utils import get_logger

from ..models.mlp import TimeEmbeddingMLP
from ..pydantic import YamlBaseModel
from ..jobs import BaseJob
from .dataset import SpiralsDataModule


DEFAULT_TRAINING_JOB_METRICS = {
    'train': {},
    'val': {},
    'test': {},
    'predict': {}
}


app = typer.Typer(
    help="Commands for Neural ODE generative models."
)


class MLPSpiralFit(BaseJob, YamlBaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Annotated[
        TimeEmbeddingMLP,
        BeforeValidator(TimeEmbeddingMLP.pydantic_before_validator),
        PlainSerializer(TimeEmbeddingMLP.pydantic_plain_serializer)
    ]

    accelerator: str = 'cpu'
    checkpoints_dir: Path | None = None
    datamodule: Annotated[
        SpiralsDataModule,
        BeforeValidator(SpiralsDataModule.pydantic_before_validator),
        PlainSerializer(SpiralsDataModule.pydantic_plain_serializer)
    ] = SpiralsDataModule()
    max_epochs: int = 3
    metrics: dict[str, dict[str, float]] = DEFAULT_TRAINING_JOB_METRICS.copy()
    monitor: str = 'val_mse'
    patience: int = 10

    def from_mlflow(self, mlflow_run) -> Self:
        raise NotImplementedError(
            "Loading TrainingJob from MLflow is not implemented yet.")

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
        self.model = TimeEmbeddingMLP.load_from_checkpoint(
            checkpoint.best_model_path)
        self.log_model(tmp_dir, checkpoint)
        logger.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

    def log_model(self):
        raise NotImplementedError("Logging the model is not implemented yet.")


# @app.command()
# def mlp_spiral_fit(config_path: ConfigOption) -> MLPSpiralFit:
#     job = MLPSpiralFit.from_yaml(config_path)
#     job.run()
#     return job


class MLP(BaseJob, YamlBaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Annotated[
        TimeEmbeddingMLP,
        BeforeValidator(TimeEmbeddingMLP.pydantic_before_validator),
        PlainSerializer(TimeEmbeddingMLP.pydantic_plain_serializer)
    ]

    accelerator: str = 'cpu'
    checkpoints_dir: Path | None = None
    datamodule: Annotated[
        SpiralsDataModule,
        BeforeValidator(SpiralsDataModule.pydantic_before_validator),
        PlainSerializer(SpiralsDataModule.pydantic_plain_serializer)
    ] = SpiralsDataModule()
    max_epochs: int = 3
    metrics: dict[str, dict[str, float]] = DEFAULT_TRAINING_JOB_METRICS.copy()
    monitor: str = 'val_mse'
    patience: int = 10

    def from_mlflow(self, mlflow_run) -> Self:
        raise NotImplementedError(
            "Loading TrainingJob from MLflow is not implemented yet.")

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
        self.model = TimeEmbeddingMLP.load_from_checkpoint(
            checkpoint.best_model_path)
        self.log_model(tmp_dir, checkpoint)
        logger.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

    def log_model(self):
        raise NotImplementedError("Logging the model is not implemented yet.")
