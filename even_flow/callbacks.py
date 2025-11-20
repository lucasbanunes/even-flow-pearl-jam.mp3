import lightning as L
import torch


class MetricHistoryCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def _update_history(self, trainer, metrics):
        # Only accumulate metrics logged at the epoch end
        for key, val in metrics.items():
            if key not in self.history:
                self.history[key] = []
            # Ensure the value is a standard Python type (float) before appending
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.history[key].append(val)

    def on_train_epoch_end(self, trainer, pl_module):
        # Access logged metrics via trainer.logged_metrics
        train_metrics = trainer.logged_metrics
        # Filter for metrics relevant to training (e.g., "train_loss_epoch")
        train_metrics = {k: v for k, v in train_metrics.items()
                         if 'train_' in k or 'loss' in k}
        self._update_history(trainer, train_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Access logged metrics via trainer.logged_metrics
        val_metrics = trainer.logged_metrics
        # Filter for metrics relevant to validation (e.g., "val_acc_epoch")
        val_metrics = {k: v for k, v in val_metrics.items() if 'val_' in k}
        self._update_history(trainer, val_metrics)
