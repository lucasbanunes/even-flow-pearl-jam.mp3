from abc import ABC, abstractmethod
from typing import Generator, Self
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import mlflow


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
    def to_mlflow(self, prefix: str = '') -> None:
        raise NotImplementedError(
            "to_mlflow method must be implemented by subclasses.")

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
        return instance