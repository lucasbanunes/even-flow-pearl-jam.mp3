from typing import Literal, Optional, Any
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import BinaryROC


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


def sp_index_pytorch(tpr: torch.Tensor, fpr: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Specificity-Positive index (SP) given true positive rate (TPR)
    and false positive rate (FPR) using PyTorch.

    Parameters
    ----------
    tpr : torch.Tensor
        True Positive Rate (TPR)
    fpr : torch.Tensor
        False Positive Rate (FPR)

    Returns
    -------
    torch.Tensor
        Specificity-Positive index (SP)
    """
    return torch.sqrt(
        torch.sqrt(tpr * (1 - fpr)) *
        ((tpr + (1 - fpr)) / 2)
    )


class MaxSPMetrics(BinaryROC):

    def __init__(self,
                 thresholds: int | list[float] | torch.Tensor | None = None,
                 ignore_index: Optional[int] = None,
                 validate_args: bool = True,
                 **kwargs: Any,
                 ) -> None:
        super().__init__(thresholds, ignore_index, validate_args, **kwargs)
        self.add_state("negatives", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("positives", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds, target)
        self.negatives += torch.sum(target == 0)
        self.positives += torch.sum(target == 1)

    def compute(self):
        fpr, tpr, thresh = super().compute()
        auc = torch.trapezoid(tpr, fpr)
        sp = sp_index_pytorch(tpr, fpr)
        max_sp_index = sp.argmax()
        sp = sp[max_sp_index]
        fpr = fpr[max_sp_index]
        tpr = tpr[max_sp_index]
        tp = tpr * self.positives
        tn = (1 - fpr) * self.negatives
        fp = fpr * self.negatives
        fn = (1 - tpr) * self.positives
        thresh = thresh[max_sp_index]
        acc = (tp + tn) / (tp + tn + fp + fn)
        return acc, sp, auc, fpr, tpr, tp, tn, fp, fn, thresh

    def compute_arrays(self):
        fpr, tpr, thresholds = super().compute()
        sp = sp_index_pytorch(tpr, fpr)
        tp = tpr * self.positives
        tn = (1 - fpr) * self.negatives
        fp = fpr * self.negatives
        fn = (1 - tpr) * self.positives
        acc = (tp + tn) / (tp + tn + fp + fn)
        return {
            'acc': acc,
            'sp': sp,
            'fpr': fpr,
            'tpr': tpr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'thresholds': thresholds
        }
