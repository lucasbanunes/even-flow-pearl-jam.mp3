from typing import Generator
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
