from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Type
import mlflow
from mlflow.entities import Run
from mlflow.models.model import ModelInfo
import torch
import json

from .dataset import MNISTDataset, MNIST_MEAN, MNIST_STD
from ..utils import get_logger
from ..jobs import BaseJob, DEFAULT_TRAINING_JOB_METRICS
from ..pydantic import MLFlowLoggedModel, YamlBaseModel
from ..models.cnf import ZukoCNFModel
from ..models.real_nvp import RealNVPModel
from ..mlflow import load_json


class BaseMNISTJob(BaseJob, YamlBaseModel):

    MODEL_PREFIX: ClassVar[str] = 'model'
    DATASET_PREFIX: ClassVar[str] = 'dataset'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'

    dataset: MNISTDataset = MNISTDataset()
    model: MLFlowLoggedModel
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def _run(self, tmp_dir: Path, active_run: Run):
        logger = get_logger()

        self.model.to_mlflow(prefix=self.MODEL_PREFIX)
        self.dataset.to_mlflow(prefix=self.DATASET_PREFIX)
        logger.info('Fitting model...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        datamodule = self.dataset.as_lightning_datamodule()
        fit_response = self.model.fit(
            datamodule,
            prefix=self.MODEL_PREFIX)
        if isinstance(fit_response, ModelInfo):
            model_info = fit_response
        elif len(fit_response) == 1:
            model_info = fit_response[0]
        elif len(fit_response) == 2:
            model_info = fit_response[1]
        else:
            raise ValueError(
                'Unexpected fit response from model.fit() method.')
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds(),
            model_id=model_info.model_id
        )

        dataloaders = {
            'train': datamodule.train_dataloader(),
        }
        try:
            dataloaders['val'] = datamodule.val_dataloader()
        except Exception as e:
            if not str(e).startswith('`val_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['test'] = datamodule.test_dataloader()
        except Exception as e:
            if not str(e).startswith('`test_dataloader` must be implemented'):
                raise e
        try:
            dataloaders['predict'] = datamodule.predict_dataloader()
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

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix=''):
        kwargs = {}
        kwargs['dataset'] = MNISTDataset.from_mlflow(
            mlflow_run, prefix=cls.DATASET_PREFIX)
        model_type: Type[MLFlowLoggedModel] = cls.model_fields['model'].annotation
        kwargs['model'] = model_type.from_mlflow(
            mlflow_run, prefix=cls.MODEL_PREFIX)
        kwargs['metrics'] = load_json(mlflow_run.info.run_id,
                                      cls.METRICS_ARTIFACT_PATH)
        return cls(**kwargs)


class MNISTZukoCNFModel(ZukoCNFModel):

    def sample(self, shape: tuple[int]) -> tuple[torch.Tensor, torch.Tensor]:
        base, transformed = super().sample(shape)
        base = base.reshape((shape[0], 1, 28, 28))
        transformed = transformed.reshape((shape[0], 1, 28, 28))
        return base, (transformed * MNIST_STD) + MNIST_MEAN


class MNISTZukoCNFJob(BaseMNISTJob):

    model: MNISTZukoCNFModel


class MNISTRealNVPModel(RealNVPModel):

    def sample(self, shape: tuple[int]) -> tuple[torch.Tensor, torch.Tensor]:
        base, transformed = super().sample(shape)
        base = base.reshape((shape[0], 1, 28, 28))
        transformed = transformed.reshape((shape[0], 1, 28, 28))
        return base, (transformed * MNIST_STD) + MNIST_MEAN


class MNISTRealNVPJob(BaseMNISTJob):

    model: MNISTRealNVPModel
