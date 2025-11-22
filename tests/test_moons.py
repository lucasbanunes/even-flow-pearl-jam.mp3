from pathlib import Path
from even_flow.moons.jobs import time_embedding_neural_ode


def test_time_embedding_neural_ode(data_dir: Path):
    config = data_dir / 'test_time_embedding_neural_ode.yaml'
    time_embedding_neural_ode(config)
