from typing import Generator, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import mlflow
import json


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
