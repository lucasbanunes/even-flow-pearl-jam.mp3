from even_flow.utils import set_logger
from even_flow.mnist.dataset import MNISTDataset
from even_flow.mnist.jobs import (
    MNISTRealNVPJob,
    MNISTRealNVPModel,
    MNISTZukoCNFJob,
    MNISTZukoCNFModel
)
from itertools import product
import mlflow
import warnings
import submitit
from pathlib import Path
from datetime import datetime
import typer
warnings.filterwarnings("ignore")


logger = set_logger()

dataset = MNISTDataset(
    flatten=True
)

app = typer.Typer()


def run_job(job, mlflow_experiment_name):
    mlflow.set_experiment(mlflow_experiment_name)
    job.run()


@app.command()
def main(debug: bool = True):
    jobs_to_run = []
    max_epochs = 1
    neuron_options = [
        [16, 16],
        [16, 16, 16, 16],
        [64, 64],
        [64, 64, 64, 64],
        [256, 256]
    ]
    activation_options = ['gelu', 'tanh']
    learning_rate = 1e-3

    experiment_name = "MNIST RealNVP"

    for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
        job = MNISTRealNVPJob(
            name=f'realnvp-mnist-{i}',
            dataset=dataset,
            model=MNISTRealNVPModel(
                features=28*28,
                transforms=4,
                hidden_features=neurons,
                max_epochs=max_epochs,
                activation=activation,
                checkpoint=dict(
                    monitor='val_loss',
                    mode='min',
                ),
                early_stopping=dict(
                    monitor='val_loss',
                    mode='min',
                    patience=5,
                    min_delta=1e-3,
                    stopping_threshold=-10
                ),
                learning_rate=learning_rate,
                accelerator='cpu',
                enable_progress_bar=False
            )
        )
        jobs_to_run.append((job, experiment_name))

    experiment_name = 'Moons Zuko Hutchinson CNF'

    for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
        job = MNISTZukoCNFJob(
            name=f'real-nvp-moons-{i}',
            dataset=dataset,
            model=MNISTZukoCNFModel(
                features=28*28,
                hidden_features=neurons,
                checkpoint=dict(
                    monitor='val_loss',
                    mode='min',
                ),
                early_stopping=dict(
                    monitor='val_loss',
                    mode='min',
                    patience=5,
                    min_delta=1e-2,
                    stopping_threshold=-10
                ),
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                activation=activation,
                accelerator='cpu',
                enable_progress_bar=False
            )
        )
        jobs_to_run.append((job, experiment_name))

    logs_dir = Path.home() / 'logs' / \
        f'run_moons_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    if debug:
        executor = submitit.DebugExecutor(folder=logs_dir)
    else:
        executor = submitit.AutoExecutor(folder=logs_dir)

    executor.update_parameters(slurm_array_parallelism=2,
                               timeout_min=3*60,
                               cpus_per_task=8,
                               slurm_partition="gpu")
    with executor.batch():
        for job, mlflow_experiment_name in jobs_to_run:
            logger.info(f'Submitting job: {job.name}')
            executor.submit(run_job, job, mlflow_experiment_name)


if __name__ == "__main__":
    app()
