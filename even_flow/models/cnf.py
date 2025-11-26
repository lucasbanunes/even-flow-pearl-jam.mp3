from functools import partial
from typing import Literal
import torch
from torch import nn
from torch.func import vmap
from torchdiffeq import odeint, odeint_adjoint
from torch.distributions import Distribution

from ..torch import memory_optimized_divergence1d


type PossibleStrategies = Literal['memory_optimized', 'speed_optimized']


class CNF1D(nn.Module):

    def __init__(self,
                 vector_field: nn.Module,
                 adjoint: bool = True,
                 base_distribution: Distribution | None = None,
                 integration_times: torch.Tensor | None = None,
                 solver: str = 'dopri5',
                 atol: float = 1e-5,
                 rtol: float = 1e-5,
                 divergence_strategy: PossibleStrategies = 'memory_optimized'
                 ):
        super().__init__()
        self.vector_field = vector_field
        self.adjoint = adjoint
        if self.adjoint:
            self.odeint_func = odeint_adjoint
        else:
            self.odeint_func = odeint
        self.base_distribution = base_distribution

        if integration_times is None:
            self.integration_times = torch.Tensor(
                [0.0, 1.0], dtype=torch.float32)
        else:
            self.integration_times = integration_times

        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        self.divergence_strategy = divergence_strategy
        if self.divergence_strategy == 'memory_optimized':
            self.divergence_strategy = memory_optimized_divergence1d
        else:
            raise ValueError(
                f"Unsupported divergence strategy: {self.divergence_strategy}")

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        div0 = torch.zeros_like(z0)
        z, int_div = self.odeint_func(
            self.augmented_function,
            (z0, div0),
            self.integration_times.type_as(z0),
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return z[-1], int_div[-1]

    def augmented_function(self, t, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = state
        fixed_t = partial(self.vector_field, t)
        divergence_func = self.divergence_strategy(fixed_t)
        vectorized_dvergence_func = vmap(divergence_func)
        z_eval, divergence = vectorized_dvergence_func(z)
        return z_eval, divergence

    def log_prob(self, z0: torch.Tensor) -> torch.Tensor:
        zf, div_int = self.forward(z0)
        logp_zf = self.base_distribution.log_prob(zf)
        return logp_zf + div_int
