from typing import Annotated, ClassVar, Type
from pydantic import Field
import torch
from zuko.flows import RealNVP
import lightning as L
from torchmetrics import MetricCollection, MeanMetric
import mlflow
from mlflow.entities import Run
import json

from ..torch import ModuleNameType, TORCH_MODULES


from .lightning import (
    LearningRateType,
    LightningModel
)

type HiddenFeaturesType = Annotated[
    list[int],
    Field(
        description="List of hidden features for the coupling layers' neural networks."
    )
]

type ActivationType = Annotated[
    ModuleNameType,
    Field(
        description="Activation function name for the coupling layers' neural networks."
    )
]


class RealNVPModule(L.LightningModule):
    def __init__(self, model_config: 'RealNVPModel'):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = RealNVP(
            features=model_config.features,
            context=model_config.context,
            transforms=model_config.transforms,
            randmask=model_config.randmask,
            hidden_features=model_config.hidden_features,
            activation=TORCH_MODULES.get(model_config.activation, None),
        )
        self.learning_rate = model_config.learning_rate

        self.train_metrics = MetricCollection({
            'loss': MeanMetric()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

    def forward(self, x, context=None):
        log_prob = self.model(context).log_prob(x).reshape(-1, 1)
        return log_prob

    def training_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.train_metrics.update(loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.val_metrics.update(loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        log_prob = self.forward(batch[0])
        loss = -log_prob.mean()
        self.test_metrics.update(loss)
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        metric_values = self.test_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"test_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

    def get_test_metrics(self) -> dict[str, float]:
        return self.test_metrics.compute()

    def get_val_metrics(self) -> dict[str, float]:
        return self.val_metrics.compute()

    def get_train_metrics(self) -> dict[str, float]:
        return self.train_metrics.compute()


class RealNVPModel(LightningModel):

    LIGHTNING_MODULE_ARTIFACT_PATH: ClassVar[str] = 'real_nvp.ckpt'
    LIGHTNING_MODULE_TYPE: ClassVar[Type[L.LightningModule]] = RealNVPModule

    features: int

    context: int = 0
    transforms: int = 3
    randmask: bool = False
    learning_rate: LearningRateType = 1e-3
    hidden_features: HiddenFeaturesType
    activation: ActivationType | None = None

    lightning_module: Annotated[
        RealNVPModule | None,
        Field(
            description="The Lightning module associated with the model.",
        )
    ] = None

    def get_new_lightning_module(self):
        return RealNVPModule(model_config=self)

    def get_dist(self, context=None):
        return self.lightning_module.model(context)

    def _to_mlflow(self, prefix=''):
        super()._to_mlflow(prefix=prefix)
        if prefix:
            prefix += '.'
        mlflow.log_param(f"{prefix}features", self.features)
        mlflow.log_param(f"{prefix}context", self.context)
        mlflow.log_param(f"{prefix}transforms", self.transforms)
        mlflow.log_param(f"{prefix}randmask", self.randmask)
        mlflow.log_param(f"{prefix}learning_rate", self.learning_rate)
        mlflow.log_param(f"{prefix}hidden_features",
                         json.dumps(self.hidden_features))
        mlflow.log_param(f"{prefix}activation", self.activation)

    @classmethod
    def _from_mlflow(cls,
                     mlflow_run: Run,
                     prefix='', **kwargs):
        kwargs = super()._from_mlflow(mlflow_run, prefix=prefix, **kwargs)
        if prefix:
            prefix += '.'
        kwargs['features'] = int(
            mlflow_run.data.params[f'{prefix}features'])
        kwargs['context'] = int(
            mlflow_run.data.params.get(f'{prefix}context',
                                       cls.model_fields['context'].default))
        kwargs['transforms'] = int(
            mlflow_run.data.params.get(f'{prefix}transforms',
                                       cls.model_fields['transforms'].default))
        kwargs['randmask'] = mlflow_run.data.params.get(f'{prefix}randmask',
                                                        cls.model_fields['randmask'].default)
        kwargs['randmask'] = kwargs['randmask'] in ['True', 'true', True]
        kwargs['learning_rate'] = float(
            mlflow_run.data.params.get(f'{prefix}learning_rate',
                                       cls.model_fields['learning_rate'].default))
        hidden_features_str = mlflow_run.data.params[f'{prefix}hidden_features']
        kwargs['hidden_features'] = json.loads(hidden_features_str)
        activation_str = mlflow_run.data.params.get(
            f'{prefix}activation', cls.model_fields['activation'].default
        )
        kwargs['activation'] = activation_str if activation_str != 'None' else None
        instance = cls(**kwargs)
        return instance

    def sample(self, shape: tuple[int],
               context: torch.Tensor | None = None) -> torch.Tensor:
        normalizing_flow = self.lightning_module.model(context)
        if normalizing_flow.base.has_rsample:
            z = normalizing_flow.base.rsample(shape)
        else:
            z = normalizing_flow.base.sample(shape)

        return z, normalizing_flow.transform.inv(z)
