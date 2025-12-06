import logging
import subprocess
from pathlib import Path
import mlflow
import torch


from even_flow.mnist.cli import (
    zuko_cnf,
    real_nvp
)
from even_flow.mnist.jobs import (
    MNISTZukoCNFJob,
    MNISTRealNVPJob
)


def test_mnist_real_nvp(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_mnist_real_nvp")

    config = data_dir / 'test_mnist' / 'test_real_nvp.yaml'
    job = real_nvp(config)
    loaded_job: MNISTRealNVPJob = MNISTRealNVPJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump(exclude='model.lightning_module')
    loaded_job_dict = loaded_job.model_dump(exclude='model.lightning_module')
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job"
    assert loaded_job.model.lightning_module is not None, \
        "The lightning module was not properly initialized upon loading."

    base, transformed = loaded_job.model.sample((10,))
    assert base.shape == (10, 1, 28, 28), "Sampled base shape is incorrect."
    assert transformed.shape == (
        10, 1, 28, 28), "Sampled transformed shape is incorrect."
    assert not torch.allclose(
        base, transformed), "Base distribution and transformed distribution should not be the same."


def test_cli_mnist_real_nvp(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_mnist_real_nvp")

    config = data_dir / 'test_mnist' / 'test_real_nvp.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'mnist',
         'real-nvp',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."


def test_mnist_zuko_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_mnist_zuko_cnf")

    config = data_dir / 'test_mnist' / 'test_zuko_cnf.yaml'
    job = zuko_cnf(config)
    loaded_job: MNISTZukoCNFJob = MNISTZukoCNFJob.from_mlflow_run_id(
        job.id_
    )
    job_dict = job.model_dump(exclude='model.lightning_module')
    loaded_job_dict = loaded_job.model_dump(exclude='model.lightning_module')
    logging.info(f"Original job: {job_dict}")
    logging.info(f"Loaded job: {loaded_job_dict}")
    assert job_dict == loaded_job_dict, "The loaded job does not match the original job."

    assert loaded_job.model.lightning_module is not None, \
        "The lightning module was not properly initialized upon loading."

    base, transformed = loaded_job.model.sample((10,))
    assert base.shape == (10, 1, 28, 28), "Sampled base shape is incorrect."
    assert transformed.shape == (
        10, 1, 28, 28), "Sampled transformed shape is incorrect."
    assert not torch.allclose(
        base, transformed), "Base distribution and transformed distribution should not be the same."


def test_cli_mnist_zuko_cnf(data_dir: Path):

    mlflow.set_experiment(f"{__name__}_test_cli_mnist_zuko_cnf")

    config = data_dir / 'test_mnist' / 'test_zuko_cnf.yaml'
    result = subprocess.run(
        ['uv', 'run', 'python', 'cli.py',
         'mnist',
         'zuko-cnf',
         '--config', str(config)],
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT:\n{result.stdout}")
    logging.info(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, "CLI command failed."
