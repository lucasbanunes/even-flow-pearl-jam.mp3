"""Neural ODE training module.

This module provides :class:`NeuralODE`, a LightningModule wrapper around a
parameterized vector field that can be integrated with torchdiffeq. It
supports using the adjoint method for memory-efficient gradient computation,
configurable solvers/tolerances, and basic metric logging for training,
validation and testing.

Docstrings follow NumPy style and document the public API used by the rest of
the repository.
"""

import torch
import lightning as L
from torchdiffeq import odeint_adjoint, odeint
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError


class NeuralODE(L.LightningModule):
    """LightningModule wrapping a neural ODE for regression tasks.

    The module integrates a provided ``vector_field`` network over time using
    ``torchdiffeq`` and exposes training/validation/test steps compatible with
    PyTorch Lightning. It logs mean squared error (MSE) metrics and supports
    using the adjoint method to compute gradients.

    Parameters
    ----------
    vector_field : torch.nn.Module
        Callable module implementing the vector field f(t, z) or f(z, t)
        depending on the vector field implementation. It should accept the
        state and time tensors required by ``torchdiffeq.odeint``.
    adjoint : bool, optional
        Whether to use the adjoint method for gradient computation
        (``odeint_adjoint``). Defaults to ``True``.
    solver : str, optional
        The ODE solver method passed to ``odeint``/``odeint_adjoint``
        (e.g. ``'dopri5'``). Defaults to ``'dopri5'``.
    atol : float, optional
        Absolute tolerance for the ODE solver. Defaults to ``1e-6``.
    rtol : float, optional
        Relative tolerance for the ODE solver. Defaults to ``1e-6``.

    Attributes
    ----------
    vector_field : torch.nn.Module
        The provided vector field module.
    odeint : callable
        Either ``torchdiffeq.odeint`` or ``torchdiffeq.odeint_adjoint`` depending
        on the ``adjoint`` flag.
    loss_function : torch.nn.Module
        Loss used for training (MSE).
    train_metrics, val_metrics, test_metrics : MetricCollection
        Metric collections used to accumulate and compute MSE during
        training/validation/testing.
    """

    def __init__(self,
                 vector_field: nn.Module,
                 adjoint: bool = True,
                 solver: str = 'dopri5',
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 learning_rate: float = 1e-3):
        super(NeuralODE, self).__init__()
        self.save_hyperparameters()

        self.vector_field = vector_field
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.learning_rate = learning_rate
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
        """Reset all metric collections.

        Useful to clear accumulated state between epochs or runs. This resets
        training, validation and test metric collections.
        """
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integrate the vector field starting from initial state ``z0``.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial state tensor passed to the ODE integrator. Typical shape
            is ``(batch_size, state_dim)`` or any shape accepted by the
            ``vector_field`` and ``odeint``.
        t : torch.Tensor
            Time points tensor at which to evaluate the solution. Shape is
            typically ``(n_time_points,)`` or ``(n_time_points, 1)`` depending
            on the convention used by the vector field.

        Returns
        -------
        torch.Tensor
            Tensor containing the integrated trajectory evaluated at the
            provided time points. Shape depends on ``odeint`` behavior and the
            input shapes (commonly ``(n_time_points, batch_size, state_dim)``).
        """
        zt = self.odeint(self.vector_field, z0, t, method=self.solver,
                         atol=self.atol, rtol=self.rtol)
        return zt

    def training_step(self,
                      batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        """Perform a training step and log training loss.

        Parameters
        ----------
        batch : tuple
            Tuple containing ``(t, z, z_true)`` where ``t`` are the time
            points, ``z`` is the input state (initial or batched), and
            ``z_true`` is the ground-truth trajectory to match.
        batch_idx : int
            Index of the current batch (unused but required by Lightning).

        Returns
        -------
        torch.Tensor
            The computed loss for the batch used by Lightning for optimization.
        """
        t, z, z_true = batch
        z_pred = self(z, t)
        loss = self.loss_function(z_pred, z_true)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Hook called at the end of training epoch.

        Resets accumulated training metrics so each epoch starts fresh.
        """
        # Reset metrics after each epoch
        self.train_metrics.reset()

    def validation_step(self,
                        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        """Validation step: compute predictions and update validation metrics.

        Parameters
        ----------
        batch : tuple
            Tuple containing ``(t, z, z_true)`` similar to the training step.
        batch_idx : int
            Index of the current batch (unused but required by Lightning).
        """
        t, z, z_true = batch
        z_pred = self(z, t)
        self.val_metrics.update(z_pred, z_true)

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end, then reset.

        Logs each metric in ``val_metrics`` with the prefix ``val_``.
        """
        metric_values = self.val_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"val_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.val_metrics.reset()

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  batch_idx: torch.Tensor):
        """Test step: compute predictions and update test metrics.

        Parameters
        ----------
        batch : tuple
            Tuple containing ``(t, z, z_true)`` similar to training/validation.
        batch_idx : int
            Index of the current batch (unused but required by Lightning).
        """
        t, z, z_true = batch
        z_pred = self(z, t)
        self.test_metrics.update(z_pred, z_true)

    def on_test_epoch_end(self):
        """Compute and log test metrics at epoch end, then reset."""
        metric_values = self.test_metrics.compute()
        for name, value in metric_values.items():
            self.log(f"test_{name}", value, prog_bar=True)
        # Reset metrics after each epoch
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure and return the optimizer.

        The method uses Adam.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer
