from pydantic import BaseModel, ConfigDict

from .models import NeuralODE


DEFAULT_TRAINING_JOB_METRICS = {
    'train': {},
    'val': {},
    'test': {},
    'predict': {}
}


class TrainingJob(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: NeuralODE | None
    metrics: dict[str, dict[str, float]] = DEFAULT_TRAINING_JOB_METRICS.copy()

    def from_mlflow(self, mlflow_run) -> "TrainingJob":
        raise NotImplementedError(
            "Loading TrainingJob from MLflow is not implemented yet.")

    def run(self):
        raise NotImplementedError(
            "Running the TrainingJob is not implemented yet.")

    def log_model(self):
        raise NotImplementedError("Logging the model is not implemented yet.")
