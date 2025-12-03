from typing import Annotated, ClassVar, Type
from datetime import datetime, timezone
from functools import cached_property
from abc import ABC, abstractmethod

from tqdm import tqdm, trange
import mlflow
from mlflow.models.model import ModelInfo
import torch
from torch import nn
from torch.utils.data import DataLoader
from pydantic import PrivateAttr
from ..utils import get_logger
from ..pydantic import MLFlowLoggedModel
from .lightning import (
    TrainerAcceleratorType,
    ProfilerType,
    MaxEpochsType,
    TrainerTestVerbosityType,
    EarlyStoppingConfig,
)


class BaseNNModule(nn.Module, ABC):

    @abstractmethod
    def training_step(self,
                      batch: tuple[torch.Tensor, ...],
                      batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self,
                        batch: tuple[torch.Tensor, ...],
                        batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def test_step(self,
                  batch: tuple[torch.Tensor, ...],
                  batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass


class TorchModel(MLFlowLoggedModel):

    TORCH_MODEL_TYPE: ClassVar[Type[BaseNNModule]] = BaseNNModule
    TORCH_MODEL_ARTIFACT_PATH: ClassVar[str] = "model.pt"

    accelerator: TrainerAcceleratorType = 'cpu'
    profiler: ProfilerType = 'simple'
    max_epochs: MaxEpochsType = 3
    verbose: TrainerTestVerbosityType = True
    num_sanity_val_steps: int = 5

    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()

    _torch_module: Annotated[
        nn.Module | None,
        PrivateAttr()
    ] = None

    @property
    def torch_module(self) -> BaseNNModule:
        if self._torch_module is not None:
            return self._torch_module

        if self.id_:
            logger = get_logger()
            logger.debug('Loading Torch module from checkpoint...')
            self._torch_module = self.load_torch_module_from_checkpoint()
            return self._torch_module

        self._torch_module = self.TORCH_MODEL_TYPE(
            model_config=self)
        return self._torch_module

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
            train_dataloader: DataLoader,
            val_dataloader: DataLoader | None = None,
            prefix: str = '') -> ModelInfo:
        logger = get_logger()
        fit_start = datetime.now(timezone.utc)
        logger.debug('Starting training process...')

        optimizer = self.torch_module.configure_optimizers()
        current_step = 0
        for epoch in trange(self.max_epochs, desc='Epochs'):
            train_dataloader_iterator = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
            )
            val_dataloader_iterator = tqdm(
                enumerate(val_dataloader),
                total=len(val_dataloader),
            ) if val_dataloader is not None else None
            for batch_index, batch in train_dataloader_iterator:

                current_step += 1
                loss = self.torch_module.training_step(batch, batch_index)
                mlflow.log_metric(
                    f"{prefix}train_loss",
                    loss.item(),
                    step=current_step
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if val_dataloader_iterator is None:
                continue

            for batch_index, batch in val_dataloader_iterator:
                loss = self.torch_module.validation_step(batch, batch_index)
                mlflow.log_metric(
                    f"{prefix}val_loss",
                    loss.item(),
                    step=current_step
                )

        logger.debug('Training completed.')
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())

        model_info = mlflow.pytorch.log_model(
            pytorch_model=self.lightning_module,
            name=self._get_mlflow_model_name(prefix))

        return model_info
