from tempfile import TemporaryDirectory
from typing import Annotated, ClassVar, Type, Any, Self
from datetime import datetime, timezone
from functools import cached_property
from abc import abstractmethod
from pathlib import Path
import lightning as L
from tqdm import tqdm, trange
import mlflow
from mlflow.models.model import ModelInfo
import torch
from torch import nn
from pydantic import Field

from ..utils import get_logger
from ..pydantic import MLFlowLoggedModel
from .lightning import (
    MaxEpochsType,
    TrainerTestVerbosityType,
    EarlyStopping,
)
from ..mlflow import tmp_artifact_download


class BaseNNModule(nn.Module):

    def training_step(self,
                      batch: tuple[torch.Tensor, ...],
                      batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "The training_step method must be implemented by the subclass."
        )

    def validation_step(self,
                        batch: tuple[torch.Tensor, ...],
                        batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "The validation_step method must be implemented by the subclass."
        )

    def test_step(self,
                  batch: tuple[torch.Tensor, ...],
                  batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "The test_step method must be implemented by the subclass."
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError(
            "The configure_optimizers method must be implemented by the subclass."
        )

    def reset_metrics(self):
        raise NotImplementedError(
            "The reset_metrics method must be implemented by the subclass."
        )

    def get_test_metrics(self):
        raise NotImplementedError(
            "The get_test_metrics method must be implemented by the subclass."
        )


def get_torch_module_artifact_name(suffix: str, prefix='') -> str:
    if prefix and prefix.endswith('.'):
        prefix = prefix.replace('.', '_')
    elif prefix:
        prefix = prefix.replace('.', '_') + '_'
    artifact_name = f'{prefix}{suffix}'
    return artifact_name


class TorchModel(MLFlowLoggedModel):

    TORCH_MODEL_TYPE: ClassVar[Type[BaseNNModule]] = BaseNNModule
    TORCH_MODEL_ARTIFACT_PATH: ClassVar[str] = "model.pt"

    max_epochs: MaxEpochsType = 3
    verbose: TrainerTestVerbosityType = True

    early_stopping: EarlyStopping = EarlyStopping()

    torch_module: Annotated[
        nn.Module | None,
        Field(
            description="The underlying Torch module. If None, it will be created when accessed."
        )
    ] = None

    @abstractmethod
    def get_new_torch_module(self) -> BaseNNModule:
        raise NotImplementedError()

    @cached_property
    def model_name(self) -> str:
        return self.TORCH_MODEL_ARTIFACT_PATH.replace('.pt', '')

    def _get_mlflow_model_name(self, prefix: str = '') -> str:
        prefix = prefix.replace('.', '_')
        return f'{prefix}_{self.model_name}'

    def get_model_artifact_path(self, prefix: str = '') -> str:
        prefix = prefix.replace('.', '_')
        return f'{prefix}_{self.TORCH_MODEL_ARTIFACT_PATH}'

    def fit(self,
            datamodule: L.LightningDataModule,
            prefix: str = '') -> ModelInfo:
        logger = get_logger()
        fit_start = datetime.now(timezone.utc)
        logger.debug('Starting training process...')
        if self.torch_module is None:
            logger.debug('Creating new Torch module...')
            self.torch_module = self.get_new_torch_module()

        optimizer = self.torch_module.configure_optimizers()
        current_step = 0

        train_dataloader = datamodule.train_dataloader()
        val_dataloader = None
        try:
            val_dataloader = datamodule.val_dataloader()
        except Exception as e:
            if not str(e).startswith('`val_dataloader` must be implemented'):
                raise e

        with torch.enable_grad():
            for epoch in trange(self.max_epochs, desc='Epochs'):
                train_dataloader_iterator = tqdm(
                    enumerate(train_dataloader),
                    total=len(train_dataloader),
                )
                val_dataloader_iterator = tqdm(
                    enumerate(val_dataloader),
                    total=len(val_dataloader),
                ) if val_dataloader is not None else None
                epoch_start = datetime.now(timezone.utc).timestamp()
                training_loss_sum = 0
                for batch_index, batch in train_dataloader_iterator:

                    current_step += 1
                    mlflow.log_metric(
                        f"{prefix}.epoch",
                        epoch,
                        step=current_step
                    )
                    loss = self.torch_module.training_step(batch, batch_index)
                    mlflow.log_metric(
                        f"{prefix}.training_loss",
                        loss.item(),
                        step=current_step
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    training_loss_sum += loss.item()

                if val_dataloader_iterator is None:
                    continue

                for batch_index, batch in val_dataloader_iterator:
                    self.torch_module.validation_step(
                        batch, batch_index)

                epoch_end = datetime.now(timezone.utc).timestamp()
                mlflow.log_metric(
                    f"{prefix}.epoch_duration",
                    epoch_end - epoch_start,
                    step=current_step
                )
                training_loss_epoch = training_loss_sum / len(train_dataloader)
                mlflow.log_metric(
                    f"{prefix}.training_loss_epoch",
                    training_loss_epoch,
                    step=current_step
                )

        logger.debug('Training completed.')
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric(
            f"{prefix}.fit_duration", (fit_end - fit_start).total_seconds())

        logger.debug('Saving model')
        model_info = mlflow.pytorch.log_model(
            pytorch_model=self.torch_module,
            name=self._get_mlflow_model_name(prefix))

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            artifact_path = tmp_dir / get_torch_module_artifact_name(
                self.TORCH_MODEL_ARTIFACT_PATH,
                prefix
            )
            torch.save(
                self.torch_module,
                str(artifact_path)
            )
            mlflow.log_artifact(str(artifact_path))

        logger.debug('Finished fitting.')

        return model_info

    def evaluate(self,
                 dataloader: torch.utils.data.DataLoader) -> dict[str, Any]:
        for i, batch in enumerate(dataloader):
            self.torch_module.test_step(batch, i)
            metrics = self.torch_module.get_test_metrics()
            self.torch_module.reset_metrics()
        return metrics

    def _to_mlflow(self, prefix=''):
        if prefix:
            prefix += '.'
        mlflow.log_param(f"{prefix}max_epochs", self.max_epochs)
        mlflow.log_param(f"{prefix}verbose", self.verbose)
        self.early_stopping.to_mlflow(prefix=prefix + 'early_stopping')

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        unformatted_prefix = prefix
        if prefix:
            prefix += "."
        kwargs['max_epochs'] = int(mlflow_run.data.params.get(
            f'{prefix}max_epochs',
            cls.model_fields['max_epochs'].default))

        verbose_str = mlflow_run.data.params.get(
            f'{prefix}verbose',
            str(cls.model_fields['verbose'].default))
        kwargs['verbose'] = verbose_str.lower() == 'true'

        kwargs['early_stopping'] = EarlyStopping.from_mlflow(
            mlflow_run,
            prefix=prefix + 'early_stopping')
        kwargs['torch_module'] = cls.load_module_from_checkpoint(
            run_id=mlflow_run.info.run_id,
            prefix=unformatted_prefix
        )
        return kwargs

    @classmethod
    def load_module_from_checkpoint(cls, run_id: str, prefix: str = '') -> Self:
        with tmp_artifact_download(
            run_id=run_id,
            artifact_path=get_torch_module_artifact_name(
                cls.TORCH_MODEL_ARTIFACT_PATH, prefix)
        ) as artifact_path:
            return torch.load(str(artifact_path),
                              weights_only=False)
