from pathlib import Path
import pytest


@pytest.fixture
def repo_dir() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def test_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"
