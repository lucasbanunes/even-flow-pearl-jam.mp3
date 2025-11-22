from typing import Literal
import torch
import torch.nn.functional as F
from torchmetrics import Metric


class BCELogits(Metric):

    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.add_state('bce_sum', default=torch.tensor(
            0.0), dist_reduce_fx='sum')
        self.add_state('n_samples', default=torch.tensor(0),
                       dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loss = F.binary_cross_entropy_with_logits(
            preds, target.float(), reduction=self.reduction)
        self.bce_sum += loss
        self.n_samples += 1

    def compute(self) -> torch.Tensor:
        if self.reduction == 'mean':
            return self.bce_sum / self.n_samples
        elif self.reduction == 'sum':
            return self.bce_sum

        raise ValueError(f"Reduction '{self.reduction}' not supported.")
