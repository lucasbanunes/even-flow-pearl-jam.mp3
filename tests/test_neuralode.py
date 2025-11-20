from pathlib import Path
from even_flow.neuralode.jobs import mlp_spiral_fit


def test_mlp_spiral_fit(data_dir: Path):
    config = data_dir / 'test_mlp_spiral_fit.yaml'
    mlp_spiral_fit(config)
