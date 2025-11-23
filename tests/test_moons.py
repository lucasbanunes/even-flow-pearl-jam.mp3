import logging
import subprocess
from pathlib import Path
import mlflow
from even_flow.moons.cli import time_embedding_mlp_neural_ode
from even_flow.moons.jobs import MoonsTimeEmbeddinngMLPNeuralODEJob


def test_time_embedding_mlp_neural_ode(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_time_embedding_mlp_neural_ode")

    config = data_dir / 'test_time_embedding_mlp_neural_ode.yaml'
    job = time_embedding_mlp_neural_ode(config)
    loaded_job = MoonsTimeEmbeddinngMLPNeuralODEJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump()
    loaded_job_dict = loaded_job.model_dump()
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."


def test_cli_time_embedding_mlp_neural_ode(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_time_embedding_mlp_neural_ode")

    config = data_dir / 'test_time_embedding_mlp_neural_ode.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'moons',
         'time-embedding-mlp-neural-ode',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."
