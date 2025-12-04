from abc import ABC, abstractmethod
from typing import Generator, Self, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import mlflow
import json
from mlflow.entities import Run


@contextmanager
def tmp_artifact_download(run_id: str,
                          artifact_path: str) -> Generator[Path, None, None]:
    """
    Download an artifact from a run to a temporary directory.

    Parameters
    ----------
    run_id : str
        The MLFlow run ID from which to download the artifact.
    artifact_path : str
        The path to the artifact to download.

    Yields
    ------
    Path
        The path to the downloaded artifact.
    """
    with TemporaryDirectory() as tmp_dir:
        yield Path(mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=tmp_dir
        ))


class MLFlowLoggedClass(ABC):

    @abstractmethod
    def _to_mlflow(self, prefix: str = '') -> None:
        raise NotImplementedError(
            "to_mlflow method must be implemented by subclasses.")

    def to_mlflow(self, prefix: str = '') -> None:
        self._to_mlflow(prefix=prefix)
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run found.")
        self.id_ = active_run.info.run_id
        self.name = active_run.data.tags.get('mlflow.runName', None)
        self.prefix = prefix

    @classmethod
    @abstractmethod
    def _from_mlflow(cls, mlflow_run: Run,
                     prefix: str = '') -> Self:
        raise NotImplementedError(
            "from_mlflow method must be implemented by subclasses.")

    @classmethod
    def from_mlflow(cls, mlflow_run: Run,
                    prefix: str = '') -> Self:
        instance = cls._from_mlflow(mlflow_run, prefix=prefix)
        instance.id_ = mlflow_run.info.run_id
        instance.name = mlflow_run.data.tags.get('mlflow.runName', None)
        return instance

    @classmethod
    def from_mlflow_run_id(cls, run_id: str, prefix: str = '') -> Self:
        mlflow_client = mlflow.MlflowClient()
        mlflow_run = mlflow_client.get_run(run_id)
        instance = cls.from_mlflow(mlflow_run, prefix=prefix)
        return instance


def load_json(
        run_id: str,
        artifact_path: str
) -> Any:
    """
    Loads a JSON file from an MLflow run artifact.

    Parameters
    ----------
    run_id : str
        The MLflow run ID.
    artifact_path : str
        The path to the artifact in MLflow.

    Returns
    -------
    Any
        The loaded JSON object.
    """
    with tmp_artifact_download(run_id, artifact_path) as tmp_path:
        with open(tmp_path, 'r') as f:
            return json.load(f)
