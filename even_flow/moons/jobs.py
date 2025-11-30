from pathlib import Path
from typing import Annotated, Self, ClassVar, Type
from datetime import datetime, timezone
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
import mlflow
from mlflow.entities import Run
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from mlflow.models.model import ModelInfo
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from .dataset import MoonsDataset
from ..jobs import BaseJob, DEFAULT_TRAINING_JOB_METRICS
from ..utils import get_logger
from ..pydantic import YamlBaseModel
from ..mlflow import MLFlowLoggedClass, load_json
from ..models.cnf import (
    TimeEmbeddingMLPCNFModel,
    TimeEmbeddingMLPCNFHutchinsonModel
)
from ..models.real_nvp import RealNVPModel
from ..models.neuralode import TimeEmbeddingMLPNeuralODEModel
from ..plotting import quiver_plot


class BaseMoonsJob(BaseJob, YamlBaseModel):

    def plot_comparison(self, n_samples: int = 1000,
                        axes: list[Axes] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, Figure, list[Axes]]:
        z_samples, x_samples = self.model.sample((n_samples,))
        z_samples = z_samples.detach().cpu().numpy()
        x_samples = x_samples.detach().cpu().numpy()
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(17, 8))
        else:
            fig = axes[0].get_figure()
        for ax in axes:
            ax.grid(alpha=.3, linestyle='--')
            ax.set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Base Distribution Samples')
        sns.scatterplot(
            x=z_samples[:, 0].reshape(-1),
            y=z_samples[:, 1].reshape(-1),
            ax=axes[0]
        )
        axes[1].set_title('Transformed Samples')
        sns.scatterplot(
            x=x_samples[:, 0].reshape(-1),
            y=x_samples[:, 1].reshape(-1),
            ax=axes[1]
        )
        return x_samples, z_samples, fig, axes

    def _run(self, tmp_dir: Path, active_run: Run):
        logger = get_logger()

        self.model.to_mlflow(prefix=self.MODEL_PREFIX)
        self.datamodule.to_mlflow(prefix=self.DATAMODULE_PREFIX)
        logger.info('Fitting model...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        _, model_info = self.model.fit(
            self.datamodule, prefix=self.MODEL_PREFIX)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds(),
            model_id=model_info.model_id
        )

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
            mlflow.log_metric(f'{dataset}.eval_start', eval_start.timestamp(),
                              model_id=model_info.model_id)
            metrics = self.model.evaluate(loader)
            eval_end = datetime.now(timezone.utc)
            mlflow.log_metric(f'{dataset}.eval_end', eval_end.timestamp(),
                              model_id=model_info.model_id)
            mlflow.log_metric(
                f"{dataset}.eval_duration",
                (eval_end - eval_start).total_seconds(),
                model_id=model_info.model_id
            )
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = v.item()
                else:
                    metrics[k] = v
            self.metrics[dataset].update(metrics)

        self.log_metrics(tmp_dir, model_info)

    @classmethod
    def from_mlflow(cls, mlflow_run: Run, prefix: str = '') -> Self:
        if prefix:
            prefix += "."
        metrics = load_json(mlflow_run.info.run_id,
                            cls.METRICS_ARTIFACT_PATH)
        model_type: Type[MLFlowLoggedClass] = cls.model_fields['model'].annotation
        return cls(
            model=model_type.from_mlflow(
                mlflow_run,
                prefix=f'{prefix}{cls.MODEL_PREFIX}'
            ),
            datamodule=MoonsDataset.from_mlflow(mlflow_run,
                                                prefix=f'{prefix}{cls.DATAMODULE_PREFIX}'),
            metrics=metrics,
        )

    def log_metrics(self, tmp_dir: Path, model_info: ModelInfo):
        for dataset, metrics in self.metrics.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{dataset}.{metric_name}",
                                  metric_value,
                                  model_id=model_info.model_id)

        metrics_json = tmp_dir / self.METRICS_ARTIFACT_PATH
        with metrics_json.open('w') as f:
            json.dump(self.metrics, f, indent=4)
        mlflow.log_artifact(str(metrics_json))


class MoonsTimeEmbeddingMLPCNFJob(BaseMoonsJob):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'
    MODEL_PREFIX: ClassVar[str] = 'model'

    model: TimeEmbeddingMLPCNFModel
    datamodule: Annotated[
        MoonsDataset,
        BeforeValidator(MoonsDataset.pydantic_before_validator),
        PlainSerializer(MoonsDataset.pydantic_plain_serializer)
    ] = MoonsDataset()
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def vector_field_plot(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          t: float,
                          colorbar: bool = True,
                          ax: Axes = None,
                          cmap: str = 'coolwarm') -> Axes:
        if ax is None:
            ax = plt.gca()
        quiver_plot(
            x=x,
            y=y,
            vector_field=lambda t, x: -
            self.model.lightning_module.vector_field(t, x),
            t=t,
            ax=ax,
            colorbar=colorbar,
            cmap=cmap
        )
        return ax


class MoonsTimeEmbeddingMLPCNFHutchinsonJob(MoonsTimeEmbeddingMLPCNFJob):

    model: TimeEmbeddingMLPCNFHutchinsonModel


class MoonsRealNVPJob(BaseMoonsJob):

    model_config = ConfigDict(arbitrary_types_allowed=True,
                              extra='forbid')

    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'
    MODEL_PREFIX: ClassVar[str] = 'model'

    model: RealNVPModel
    datamodule: Annotated[
        MoonsDataset,
        BeforeValidator(MoonsDataset.pydantic_before_validator),
        PlainSerializer(MoonsDataset.pydantic_plain_serializer)
    ] = MoonsDataset()
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def vector_field_plot(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          ax: Axes | None = None,
                          cmap='coolwarm') -> Axes:
        if ax is None:
            ax = plt.gca()
        with torch.no_grad():
            X, Y = torch.meshgrid(x, y, indexing='ij')
            grid_points = torch.stack([X.ravel(), Y.ravel()], dim=1)
            dist = self.model.get_dist()
            x_transformed = dist.transform.inv(grid_points)
            displacement = x_transformed - grid_points
            displacement_np = displacement.cpu().numpy()
            U = displacement_np[:, 0].reshape(X.shape)
            V = displacement_np[:, 1].reshape(Y.shape)

        # Normalize for visualization
        M = np.hypot(U, V)
        q = ax.quiver(X, Y, U, V, M,
                      cmap=cmap)
        plt.colorbar(q, ax=ax, label='Vector Magnitude')
        return ax


class MoonsTimeEmbeddinngMLPNeuralODEJob(BaseMoonsJob):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    DATAMODULE_PREFIX: ClassVar[str] = 'datamodule'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'
    MODEL_PREFIX: ClassVar[str] = 'model'

    model: TimeEmbeddingMLPNeuralODEModel
    datamodule: Annotated[
        MoonsDataset,
        BeforeValidator(MoonsDataset.pydantic_before_validator),
        PlainSerializer(MoonsDataset.pydantic_plain_serializer)
    ] = MoonsDataset()
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def vector_field_plot(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          t: float,
                          colorbar: bool = True,
                          ax: Axes = None,
                          cmap: str = 'coolwarm') -> Axes:
        if ax is None:
            ax = plt.gca()
        quiver_plot(
            x=x,
            y=y,
            vector_field=lambda t, x: -
            self.model.lightning_module.vector_field(t, x),
            t=t,
            ax=ax,
            colorbar=colorbar,
            cmap=cmap
        )
        return ax
