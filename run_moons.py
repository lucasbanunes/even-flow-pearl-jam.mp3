from even_flow.utils import set_logger, get_logger
from even_flow.moons.jobs import (
    MoonsTimeEmbeddingMLPCNFJob,
)
from even_flow.moons.dataset import MoonsDataset
from even_flow.models.cnf import (
    TimeEmbeddingMLPCNFModel,
)
from even_flow.jobs import BaseJob
from itertools import product
import mlflow
import warnings
import submitit
from pathlib import Path
from datetime import datetime
import typer
from typing import Annotated, Literal
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")


def run_job(job, mlflow_experiment_name):
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(mlflow_experiment_name)
    logger = get_logger()
    logger.info(f'Running job: {job.name}')
    job.run()


logger = set_logger()
mlflow_client = mlflow.MlflowClient()

train_samples = 10000
val_samples = 1000
test_samples = 1000
noise = 0.05
batch_size = 32
random_state = 943874

dataset = MoonsDataset(
    train_samples=train_samples,
    val_samples=val_samples,
    test_samples=test_samples,
    noise=noise,
    batch_size=batch_size,
    random_state=random_state
)

app = typer.Typer()

SchedulerTypes = Literal['slurm', 'local']

type JobListType = list[tuple[BaseJob, str]]


def slurm_scheduler(
    job_list: JobListType, name: str, n_jobs: int, timeout: int,
    logs_dir: Path
) -> None:

    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(
        name=name,
        slurm_array_parallelism=n_jobs,
        timeout_min=timeout,
        cpus_per_task=8,
        slurm_partition="gpu",
        stderr_to_stdout=True
    )
    with executor.batch():
        for job, mlflow_experiment_name in job_list:
            logger.info(f'Submitting job: {job.name}')
            executor.submit(run_job, job, mlflow_experiment_name)


def local_scheduler(
    job_list: JobListType, n_jobs: int, **kwargs
) -> None:
    if n_jobs == 1:
        for job, mlflow_experiment_name in job_list:
            run_job(job, mlflow_experiment_name)
        return

    job_pool = Parallel(n_jobs=n_jobs)
    job_pool(
        delayed(run_job)(job, mlflow_experiment_name)
        for job, mlflow_experiment_name in job_list
    )


SCHEDULERS = {
    'slurm': slurm_scheduler,
    'local': local_scheduler,
}

INPUT_DIM = 2


@app.command()
def main(
    scheduler: Annotated[
        SchedulerTypes,
        typer.Option(
            "--scheduler",
            help="Scheduler to use.",
            case_sensitive=False
        )
    ],
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Name for the run. Only works with slurm scheduler."
        )
    ] = 'run_moons',
    n_jobs: Annotated[
        int,
        typer.Option(
            "--n-jobs",
            help="Number of jobs to run in parallel. Only works with slurm scheduler.",
            min=-1
        )
    ] = 6,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Timeout in minutes for each job. Only works with slurm scheduler."
        )
    ] = 12*60,
):
    jobs_to_run = []

    experiment_name = 'Moons Exact CNF'
    experiment_description = """
CNF treinado no conjunto MOONS usando Lightning como backend."""
    mlflow_client.create_experiment(
        experiment_name,
        tags={"mlflow.note.content": experiment_description}
    )

    neuron_options = [
        [16, 16],
        [16, 16, 16, 16],
        [64, 64],
        [64, 64, 64, 64],
        # [256, 256]
    ]
    activation_options = ['gelu', 'tanh']
    max_epochs = 1
    learning_rate = 1e-3

    for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
        job = MoonsTimeEmbeddingMLPCNFJob(
            name=f'exact-cnf-moons-{i}',
            dataset=dataset,
            model=TimeEmbeddingMLPCNFModel(
                vector_field=dict(
                    input_dim=INPUT_DIM,
                    freqs=3,
                    neurons_per_layer=neurons_per_layer + [INPUT_DIM],
                    activations=(len(neurons_per_layer) + 1)*[activation],
                ),
                adjoint=True,
                base_distribution='standard_normal',
                max_epochs=max_epochs,
                input_shape=(2,),
                learning_rate=learning_rate,
                early_stopping=dict(
                    monitor='val_loss',
                    mode='min',
                    patience=3,
                    min_delta=1e-2,
                    stopping_threshold=-10
                ),
                checkpoint=dict(
                    monitor='val_loss',
                    mode='min',
                ),
                max_time={
                    'hours': 11*60
                }
            )
        )
        jobs_to_run.append((job, experiment_name))

    logs_dir = Path(__file__).parent / 'logs' / \
        f'run_moons_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    SCHEDULERS[scheduler](
        job_list=jobs_to_run,
        name=name,
        n_jobs=n_jobs,
        timeout=timeout,
        logs_dir=logs_dir
    )


if __name__ == "__main__":
    app()
