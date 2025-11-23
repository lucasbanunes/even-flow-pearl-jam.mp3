from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import mlflow
from typer import Option

from .mlflow import MLFlowLoggedClass

DEFAULT_TRAINING_JOB_METRICS = {
    'train': {},
    'val': {},
    'test': {},
    'predict': {}
}

ID_TYPE_HELP = "Unique identifier for the job"
type IdType = Annotated[
    str | None,
    Field(
        description=ID_TYPE_HELP
    ),
    Option('--id', help=ID_TYPE_HELP)
]


NAME_TYPE_HELP = "Name of the job"
type NameType = Annotated[
    str,
    Field(
        description=NAME_TYPE_HELP
    ),
]

type ConfigOption = Annotated[
    Path,
    Option('--config',
           help="Path to the yaml configuration file for the job.")
]


class MLFlowBaseModel(BaseModel, MLFlowLoggedClass):
    """BaseModel with MLflow logging capabilities."""

    # MLFLOW_LOGGER_ATTRIBUTES: ClassVar[list[str] | None] = None

    id_: IdType = None
    name: NameType = None

    @abstractmethod
    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):
        raise NotImplementedError(
            "_run method must be implemented by subclasses.")

    def run(self):
        if self.id_ is not None:
            raise ValueError("Cannot run a job with a predefined id_.")
        with (mlflow.start_run(run_name=self.name) as active_run,
              TemporaryDirectory() as tmp_dir):
            self.id_ = active_run.info.run_id
            self.name = active_run.data.tags['mlflow.runName']
            exec_start = datetime.now(timezone.utc).timestamp()
            mlflow.log_metric("exec_start", exec_start)
            self._run(Path(tmp_dir), active_run)
            end_start = datetime.now(timezone.utc).timestamp()
            mlflow.log_metric('exec_end', end_start)
            mlflow.log_metric("exec_duration", end_start - exec_start)
