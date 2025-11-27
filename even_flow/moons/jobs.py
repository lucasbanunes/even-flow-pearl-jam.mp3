from pathlib import Path
from tempfile import TemporaryDirectory
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
import json
import torch
import matplotlib.pyplot as plt
import numpy as np


from .dataset import MoonsDataset
from .models import TimeEmbeddingMLPNeuralODEClassifier
from ..jobs import BaseJob, DEFAULT_TRAINING_JOB_METRICS
from ..utils import get_logger
from ..pydantic import YamlBaseModel
from ..mlflow import load_json
from ..models.cnf import TimeEmbeddingMLPCNF1DModel


class MoonsTimeEmbeddinngMLPNeuralODEJob(BaseJob, YamlBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_CKPT_PATH: ClassVar[str] = 'model.ckpt'
    MLFLOW_LOGGER_ATTRIBUTES: ClassVar[list[str] | None] = None
    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    MODEL_PREFIX: ClassVar[str] = 'model'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'
    QUIVER_PLOT_ARTIFACT_PATH: ClassVar[str] = 'quiver_plot.png'

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
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()
    patience: int = 10
    verbose: bool = True

    @classmethod
    def from_mlflow(cls, mlflow_run: Run, prefix: str = '') -> Self:
        if prefix:
            prefix += "."
        metrics = load_json(mlflow_run.info.run_id,
                            cls.METRICS_ARTIFACT_PATH)
        return cls(
            max_epochs=int(mlflow_run.data.params[f'{prefix}max_epochs']),
            model=TimeEmbeddingMLPNeuralODEClassifier.from_mlflow(
                mlflow_run,
                prefix=f'{prefix}{cls.MODEL_PREFIX}'
            ),
            monitor=mlflow_run.data.params[f'{prefix}monitor'],
            accelerator=mlflow_run.data.params.get(
                f'{prefix}accelerator', cls.model_fields['accelerator'].default),
            checkpoints_dir=mlflow_run.data.params.get(
                f'{prefix}checkpoints_dir', None),
            datamodule=MoonsDataset.from_mlflow(mlflow_run,
                                                prefix=f'{prefix}{cls.DATAMODULE_PREFIX}'),
            metrics=metrics,
            patience=int(mlflow_run.data.params.get(
                f'{prefix}patience', cls.model_fields['patience'].default)),
            verbose=bool(mlflow_run.data.params.get(
                f'{prefix}verbose', cls.model_fields['verbose'].default)),
        )

    def log_params(self, prefix: str = '') -> None:
        if prefix:
            prefix += "."
        mlflow.log_param(f'{prefix}max_epochs', self.max_epochs)
        mlflow.log_param(f'{prefix}monitor', self.monitor)
        mlflow.log_param(f'{prefix}accelerator', self.accelerator)
        if self.checkpoints_dir:
            mlflow.log_param(f'{prefix}checkpoints_dir',
                             str(self.checkpoints_dir))
        mlflow.log_param(f'{prefix}patience', self.patience)
        mlflow.log_param(f'{prefix}verbose', self.verbose)

    def to_mlflow(self, prefix: str = '') -> None:
        if prefix:
            prefix += "."
        self.log_params(prefix=prefix)
        self.model.to_mlflow(prefix=f'{prefix}{self.MODEL_PREFIX}')
        self.datamodule.to_mlflow(prefix=f'{prefix}{self.DATAMODULE_PREFIX}')
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            self.log_metrics(tmp_dir)

    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):

        logger = get_logger()

        self.log_params()
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

        dataloaders = {
            'train': self.datamodule.train_dataloader(),
        }
        try:
            dataloaders['val'] = self.datamodule.val_dataloader()
        except Exception as e:
            if not str(e).startswith('`val_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['test'] = self.datamodule.test_dataloader()
        except Exception as e:
            if not str(e).startswith('`test_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['predict'] = self.datamodule.predict_dataloader()
        except Exception as e:
            if not str(e).startswith('`predict_dataloader` must be implemented'):
                raise e

        self.model.reset_metrics()
        for dataset, loader in dataloaders.items():
            logger.info(f'Evaluating best model on {dataset} dataset')
            trainer.test(
                model=self.model,
                dataloaders=loader,
                verbose=self.verbose
            )
            self.metrics[dataset]['nfe'] = self.model.vector_field.nfe
            dataset_metrics = self.model.get_test_metrics()
            dataset_metrics = {k: v.item() for k, v in dataset_metrics.items()}
            self.metrics[dataset].update(dataset_metrics)
            self.model.reset_metrics()

        self.log_metrics(tmp_dir)

    def log_metrics(self, tmp_dir: Path):
        for dataset, metrics in self.metrics.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{dataset}.{metric_name}", metric_value)

        metrics_json = tmp_dir / self.METRICS_ARTIFACT_PATH
        with metrics_json.open('w') as f:
            json.dump(self.metrics, f, indent=4)
        mlflow.log_artifact(str(metrics_json))

    def quiver_plot(self) -> plt.Figure:
        X, Y = torch.meshgrid(torch.linspace(-1.5, 2.5, 20),
                              torch.linspace(-1.0, 1.5, 20), indexing='ij')
        grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        dxdy = self.model.vector_field.forward(torch.Tensor([0.]), grid_points)
        U = dxdy[:, 0].reshape(X.shape).detach().cpu().numpy()
        V = dxdy[:, 1].reshape(Y.shape).detach().cpu().numpy()
        M = np.hypot(U, V)

        fig, ax = plt.subplots()
        ax.grid(alpha=.3, linestyle='--')
        q = ax.quiver(X, Y, U, V, M)
        plt.colorbar(q, ax=ax, label='Vector Magnitude')
        ax.set(
            title='Learned Vector Field',
            xlabel='X',
            ylabel='Y'
        )
        fig.tight_layout()
        return fig


class MoonsTimeEmbeddingMLPCNF(BaseJob, YamlBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'
    MODEL_PREFIX: ClassVar[str] = 'model'

    model: TimeEmbeddingMLPCNF1DModel
    datamodule: Annotated[
        MoonsDataset,
        BeforeValidator(MoonsDataset.pydantic_before_validator),
        PlainSerializer(MoonsDataset.pydantic_plain_serializer)
    ] = MoonsDataset()
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def _run(self, tmp_dir: Path):
        logger = get_logger()

        self.model.to_mlflow(prefix=self.MODEL_PREFIX)
        self.datamodule.to_mlflow(prefix=self.DATAMODULE_PREFIX)
        logger.info('Fitting model...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        self.model.fit(self.datamodule, prefix=self.MODEL_PREFIX)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())

        dataloaders = {
            'train': self.datamodule.train_dataloader(),
        }
        try:
            dataloaders['val'] = self.datamodule.val_dataloader()
        except Exception as e:
            if not str(e).startswith('`val_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['test'] = self.datamodule.test_dataloader()
        except Exception as e:
            if not str(e).startswith('`test_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['predict'] = self.datamodule.predict_dataloader()
        except Exception as e:
            if not str(e).startswith('`predict_dataloader` must be implemented'):
                raise e

        for dataset, loader in dataloaders.items():
            logger.info(f'Evaluating model on {dataset} dataset')
            eval_start = datetime.now(timezone.utc)
            mlflow.log_metric(f'{dataset}.eval_start', eval_start.timestamp())
            metrics = self.model.evaluate(loader)
            eval_end = datetime.now(timezone.utc)
            mlflow.log_metric(f'{dataset}.eval_end', eval_end.timestamp())
            mlflow.log_metric(
                f"{dataset}.eval_duration",
                (eval_end - eval_start).total_seconds()
            )
            metrics = {k: v.item() for k, v in metrics.items()}
            self.metrics[dataset].update(metrics)

        self.log_metrics(tmp_dir)

    def log_metrics(self, tmp_dir: Path):
        for dataset, metrics in self.metrics.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{dataset}.{metric_name}", metric_value)

        metrics_json = tmp_dir / self.METRICS_ARTIFACT_PATH
        with metrics_json.open('w') as f:
            json.dump(self.metrics, f, indent=4)
        mlflow.log_artifact(str(metrics_json))
