from pathlib import Path
import pytest
import mlflow


@pytest.fixture
def repo_dir() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def test_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def setup_mlflow_tracking(data_dir: Path):
    mlflow.set_tracking_uri("file://" + str(data_dir / "mlruns"))
