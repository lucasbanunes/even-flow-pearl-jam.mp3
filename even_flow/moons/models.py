from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Self
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
from pydantic import Field
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy
import lightning as L
import mlflow
from mlflow.entities import Run

from ..mlflow import tmp_artifact_download, MLFlowLoggedClass
# from ..metrics import BCELogits
from ..models.mlp import ActivationsType, DimsType, TimeEmbeddingMLP


type AdjointType = Annotated[
    bool,
    Field(
        default=True,
        help="Whether to use the adjoint method for gradient computation."
    )
]

type SolverType = Annotated[
    str,
    Field(
        help="The ODE solver method passed to `odeint`/`odeint_adjoint`."
    )
]

type AToleranceType = Annotated[
    float,
    Field(
        gt=0.0,
        help="Absolute Tolerance for the ODE solver."
    )
]

type RToleranceType = Annotated[
    float,
    Field(
        gt=0.0,
        help="Relative Tolerance for the ODE solver."
    )
]

type LearningRateType = Annotated[
    float,
    Field(
        gt=0.0,
        help="Learning rate for the optimizer."
    )
]


class TimeEmbeddingMLPNeuralODEClassifier(L.LightningModule, MLFlowLoggedClass):
    """LightningModule wrapping a neural ODE with time-embedded MLP for
    classification tasks.
    """

    def __init__(self,
                 input_dims: int,
                 time_embed_dims: int,
                 time_embed_freq: float,
                 neurons_per_layer: DimsType,
                 activations: ActivationsType,
                 n_classes: int,
                 adjoint: AdjointType = True,
                 solver: SolverType = 'dopri5',
                 atol: AToleranceType = 1e-6,
                 rtol: RToleranceType = 1e-6,
                 learning_rate: float = 1e-3,):
        super().__init__()
        self.save_hyperparameters()

        self.input_dims = input_dims
        self.time_embed_dims = time_embed_dims
        self.time_embed_freq = time_embed_freq
        self.neurons_per_layer = neurons_per_layer
        self.activations = activations

        self.vector_field = TimeEmbeddingMLP(
            input_dims=input_dims,
            time_embed_dims=time_embed_dims,
            time_embed_freq=time_embed_freq,
            neurons_per_layer=neurons_per_layer,
            activations=activations
        )
        self.example_input_array = torch.randn(1, self.vector_field.input_dims)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint
        if self.adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint
        self.learning_rate = learning_rate
        if n_classes < 2:
            raise ValueError(
                f"n_classes must be at least 2 for classification, got {n_classes}.")
        self.n_classes = n_classes
        self.classification_head = nn.Linear(
            in_features=self.vector_field.output_dims,
            out_features=1 if self.n_classes == 2 else self.n_classes
        )
        self.loss_function = nn.BCEWithLogitsLoss()

        # Removed BCELogits because it was returning nans, have to check after
        self.train_metrics = MetricCollection({
            # 'loss': BCELogits(),
            'accuracy': BinaryAccuracy()
        })
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.val_metrics.clone()

        self.integration_time = torch.tensor([0, 1]).float()

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.odeint(self.vector_field, x,
                         self.integration_time.type_as(x),
                         method=self.solver,
                         atol=self.atol, rtol=self.rtol)
        y = self.classification_head(xt[-1])
        return y

    def trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the full trajectory of the neural ODE for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dims).

        Returns
        -------
        torch.Tensor
            Trajectory tensor of shape (time_steps, batch_size, output_dims).
        """
        xt = odeint(self.vector_field, x,
                    self.integration_time.type_as(x),
                    method=self.solver,
                    atol=self.atol, rtol=self.rtol)
        return xt

    def quiver(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the vector field (time derivatives) at input x."""
        t = torch.zeros(x.shape[0], 1).type_as(x)
        dxdt = self.vector_field(t, x)
        return dxdt

    def training_step(self,
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        x, y = batch
        y_pred = self(x)
        self.val_metrics.update(y_pred, y)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor):
        x, y = batch
        y_pred = self(x)
        self.val_metrics.update(y_pred, y)

    def on_test_epoch_end(self):
        """Compute and log test metrics at epoch end, then reset."""
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

    @classmethod
    def pydantic_before_validator(cls, v: Any) -> Self:
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Cannot convert {type(v)} to {cls}.")

    @staticmethod
    def pydantic_plain_serializer(v: 'TimeEmbeddingMLPNeuralODEClassifier') -> dict[str, Any]:
        return {
            "input_dims": v.input_dims,
            "time_embed_dims": v.time_embed_dims,
            "time_embed_freq": v.time_embed_freq,
            "neurons_per_layer": v.neurons_per_layer,
            "activations": v.activations,
            "n_classes": v.n_classes,
            "adjoint": v.adjoint,
            "solver": v.solver,
            "atol": v.atol,
            "rtol": v.rtol,
            "learning_rate": v.learning_rate,
        }

    def to_mlflow(self, prefix: str = "",
                  model_name: str = "model",
                  checkpoint_path: str | None = None):
        if prefix:
            prefix += "."
        mlflow.log_param(f"{prefix}input_dims", self.input_dims)
        mlflow.log_param(f"{prefix}time_embed_dims", self.time_embed_dims)
        mlflow.log_param(f"{prefix}time_embed_freq", self.time_embed_freq)
        mlflow.log_param(f"{prefix}neurons_per_layer", self.neurons_per_layer)
        mlflow.log_param(f"{prefix}activations", self.activations)
        mlflow.log_param(f"{prefix}n_classes", self.n_classes)
        mlflow.log_param(f"{prefix}adjoint", self.adjoint)
        mlflow.log_param(f"{prefix}solver", self.solver)
        mlflow.log_param(f"{prefix}atol", self.atol)
        mlflow.log_param(f"{prefix}rtol", self.rtol)
        mlflow.log_param(f"{prefix}learning_rate", self.learning_rate)
        mlflow.pytorch.log_model(
            pytorch_model=self,
            name=model_name,
        )
        if checkpoint_path is not None:
            with TemporaryDirectory() as tmp_dir:
                ckpt_path = Path(tmp_dir) / f'{model_name}.ckpt'
                shutil.copy(checkpoint_path, ckpt_path)
                mlflow.log_artifact(ckpt_path)

    @classmethod
    def from_mlflow(cls, mlflow_run: Run, prefix: str = '', model_name: str = "model") -> Self:
        with tmp_artifact_download(
            run_id=mlflow_run.info.run_id,
            artifact_path=f'{model_name}.ckpt'
        ) as ckpt_path:
            return cls.load_from_checkpoint(ckpt_path)
