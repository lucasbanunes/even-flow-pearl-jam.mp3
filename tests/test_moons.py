import logging
import subprocess
from pathlib import Path
import mlflow
import torch
from even_flow.moons.cli import (
    time_embedding_mlp_neural_ode,
    time_embedding_mlp_cnf,
    time_embedding_mlp_cnf_hutchingson,
    real_nvp, zuko_cnf
)
from even_flow.moons.jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsTimeEmbeddingMLPCNFJob,
    MoonsTimeEmbeddingMLPCNFHutchinsonJob,
    MoonsRealNVPJob, MoonsZukoCNFJob
)


def test_time_embedding_mlp_neural_ode(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_time_embedding_mlp_neural_ode")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_neural_ode.yaml'
    job = time_embedding_mlp_neural_ode(config)
    loaded_job: MoonsTimeEmbeddinngMLPNeuralODEJob = \
        MoonsTimeEmbeddinngMLPNeuralODEJob.from_mlflow_run_id(
            job.id_
        )
    job_dict = job.model_dump(exclude='model.lightning_module')
    loaded_job_dict = loaded_job.model_dump(exclude='model.lightning_module')
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."
    # Ensures that the Lightning module is loaded properly.
    loaded_job.model.lightning_module


def test_cli_time_embedding_mlp_neural_ode(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_time_embedding_mlp_neural_ode")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_neural_ode.yaml'
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


def test_time_embedding_mlp_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_time_embedding_mlp_cnf")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_cnf.yaml'
    job = time_embedding_mlp_cnf(config)
    loaded_job = MoonsTimeEmbeddingMLPCNFJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump()
    loaded_job_dict = loaded_job.model_dump()
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."


def test_cli_time_embedding_mlp_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_time_embedding_mlp_cnf")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_cnf.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'moons',
         'time-embedding-mlp-cnf',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."


def test_time_embedding_mlp_cnf_hutchingson(data_dir: Path):

    mlflow.set_experiment(
        f"{__name__}_test_time_embedding_mlp_cnf_hutchingson")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_cnf_hutchingson.yaml'
    job = time_embedding_mlp_cnf_hutchingson(config)
    loaded_job = MoonsTimeEmbeddingMLPCNFHutchinsonJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump()
    loaded_job_dict = loaded_job.model_dump()
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."


def test_cli_time_embedding_mlp_cnf_hutchingson(data_dir: Path):

    mlflow.set_experiment(
        f"{__name__}_test_cli_time_embedding_mlp_cnf_hutchingson")

    config = data_dir / 'test_moons' / 'test_time_embedding_mlp_cnf_hutchingson.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'moons',
         'time-embedding-mlp-cnf-hutchingson',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."


def test_real_nvp(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_real_nvp")

    config = data_dir / 'test_moons' / 'test_real_nvp.yaml'
    job = real_nvp(config)
    loaded_job = MoonsRealNVPJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump(exclude='model.lightning_module')
    loaded_job_dict = loaded_job.model_dump(exclude='model.lightning_module')
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."
    # Ensures that the Lightning module is loaded properly.
    assert loaded_job.model.lightning_module is not None, "The Lightning module was not loaded properly."

    base, transformed = loaded_job.model.sample(shape=(10,))
    assert base.shape == (10, 2), "Sample shape is incorrect."
    assert transformed.shape == (10, 2), "Transformed shape is incorrect."
    assert not torch.allclose(base, transformed), "Base distribution and transformed distribution should not be the same."


def test_cli_real_nvp(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_real_nvp")

    config = data_dir / 'test_moons' / 'test_real_nvp.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'moons',
         'real-nvp',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."


def test_zuko_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_zuko_cnf")

    config = data_dir / 'test_moons' / 'test_zuko_cnf.yaml'
    job = zuko_cnf(config)
    loaded_job = MoonsZukoCNFJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump(exclude='model.lightning_module')
    loaded_job_dict = loaded_job.model_dump(exclude='model.lightning_module')
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."
    # Ensures that the Lightning module is loaded properly.
    loaded_job.model.lightning_module


def test_cli_zuko_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_zuko_cnf")

    config = data_dir / 'test_moons' / 'test_zuko_cnf.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'moons',
         'zuko-cnf',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."
