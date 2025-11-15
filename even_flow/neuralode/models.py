import torch
import lightning as L
from torchdiffeq import odeint_adjoint, odeint
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError


class NeuralODE(L.LightningModule):
    def __init__(self,
                 vector_field: nn.Module,
                 adjoint: bool = True,
                 solver: str = 'dopri5',
                 atol: float = 1e-6,
                 rtol: float = 1e-6):
        super(NeuralODE, self).__init__()
        self.save_hyperparameters()

        self.vector_field = vector_field
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint
        if self.adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint
        self.loss_function = nn.MSELoss()

        self.train_metrics = MetricCollection({
            'mse': MeanSquaredError()
        })
        self.val_metrics = MetricCollection({
            'mse': MeanSquaredError()
        })
        self.test_metrics = self.val_metrics.clone()

    def reset_metrics(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

    def forward(self, z0, t):
        zt = self.odeint(self.vector_field, z0, t, method=self.solver,
                         atol=self.atol, rtol=self.rtol)
        return zt

    def training_step(self,
                      batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        t, z, z_true = batch
        z_pred = self(z, t)
        loss = self.loss_function(z_pred, z_true)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        t, z, z_true = batch
        z_pred = self(z, t)
        self.val_metrics.update(z_pred, z_true)

    def on_validation_epoch_end(self):
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor):
        t, z, z_true = batch
        z_pred = self(z, t)
        self.test_metrics.update(z_pred, z_true)

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
