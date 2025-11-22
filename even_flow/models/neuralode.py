"""Neural ODE training module.

This module provides :class:`NeuralODE`, a LightningModule wrapper around a
parameterized vector field that can be integrated with torchdiffeq. It
supports using the adjoint method for memory-efficient gradient computation,
configurable solvers/tolerances, and basic metric logging for training,
validation and testing.

Docstrings follow NumPy style and document the public API used by the rest of
the repository.
"""

from typing import Self, Any, Annotated
import torch
import lightning as L
from torchdiffeq import odeint_adjoint, odeint
import torch.nn as nn
from pydantic import Field
from torchmetrics import MetricCollection, MeanSquaredError

from ..models.mlp import build_mlp, DimsType, ActivationsType


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


class MLPNeuralODE(L.LightningModule):
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
        The provided vector field module. It should receive 2 params the time t and the state z.
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
                 dims: DimsType,
                 activations: ActivationsType,
                 adjoint: AdjointType = True,
                 solver: SolverType = 'dopri5',
                 atol: AToleranceType = 1e-6,
                 rtol: RToleranceType = 1e-6,
                 learning_rate: LearningRateType = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.dims = dims
        self.activations = activations
        self.vector_field = build_mlp(dims=self.dims,
                                      activations=self.activations)
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

    def forward(self, t: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Integrate the vector field starting from initial state ``z0``.

        Parameters
        ----------
        t : torch.Tensor
            How much time to integrate the solution for. Shape is (batch_size, time)
        z0 : torch.Tensor
            Initial state tensor passed to the ODE integrator. Typical shape
            is ``(batch_size, state_dim)`` or any shape accepted by the
            ``vector_field`` and ``odeint``.

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

    @classmethod
    def pydantic_before_validator(cls, v: Any) -> Self:
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Cannot convert {type(v)} to {cls}.")

    @staticmethod
    def pydantic_plain_serializer(v: 'MLPNeuralODE') -> dict[str, Any]:
        return {
            "dims": v.dims,
            "activations": v.activations,
            "adjoint": v.adjoint,
            "solver": v.solver,
            "atol": v.atol,
            "rtol": v.rtol,
            "learning_rate": v.learning_rate,
        }
