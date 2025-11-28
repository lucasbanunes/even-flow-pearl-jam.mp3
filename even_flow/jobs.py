from abc import abstractmethod, ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Self
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
import mlflow
from typer import Option

from .utils import get_logger

DEFAULT_TRAINING_JOB_METRICS = {
    'train': {},
    'val': {},
    'test': {},
    'predict': {}
}

type ConfigOption = Annotated[
    Path,
    Option('--config',
           help="Path to the yaml configuration file for the job.")
]

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

type RunType = Annotated[
    mlflow.entities.Run | None,
    Field(
        description="MLflow Run associated with the class.",
        exclude=True
    )
]


class BaseJob(BaseModel, ABC):
    """BaseModel with MLflow logging capabilities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id_: IdType = None
    name: NameType = None
    # mlflow_run: RunType = None

    @abstractmethod
    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):
        raise NotImplementedError(
            "_run method must be implemented by subclasses.")

    def run(self):
        logger = get_logger()
        logger.debug('Started run...')
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
        logger.debug('Finished run.')
        return self.id_

    @classmethod
    @abstractmethod
    def from_mlflow(cls, mlflow_run: mlflow.entities.Run,
                    prefix: str = '') -> Self:
        raise NotImplementedError(
            "from_mlflow method must be implemented by subclasses.")

    @classmethod
    def from_mlflow_run_id(cls, run_id: str, prefix: str = '') -> Self:
        mlflow_client = mlflow.MlflowClient()
        mlflow_run = mlflow_client.get_run(run_id)
        instance = cls.from_mlflow(mlflow_run, prefix=prefix)
        instance.id_ = run_id
        instance.name = mlflow_run.data.tags.get('mlflow.runName', None)
        # instance.mlflow_run = mlflow_run
        return instance
