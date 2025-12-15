from even_flow.utils import set_logger
from even_flow.moons.jobs import (
    MoonsTimeEmbeddingMLPCNFJob,
)
from even_flow.moons.dataset import MoonsDataset
from even_flow.models.cnf import (
    TimeEmbeddingMLPCNFModel,
)
from itertools import product
import mlflow
import warnings
import submitit
from pathlib import Path
from datetime import datetime
import typer
from typing import Annotated
warnings.filterwarnings("ignore")


def run_job(job, mlflow_experiment_name):
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(mlflow_experiment_name)
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


@app.command()
def main(
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Run in debug mode with submitit DebugExecutor."
        )
    ] = False,
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Name for the run."
        )
    ] = 'run_moons',
    n_jobs: Annotated[
        int,
        typer.Option(
            "--n-jobs",
            help="Number of jobs to run in parallel."
        )
    ] = 6,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Timeout in minutes for each job."
        )
    ] = 12*60,
):
    jobs_to_run = []

    experiment_name = 'Moons Exact CNF Corrected Div Scale'
    experiment_description = """
    CNF treinado no conjunto MOONS. Implementado com o div scale corrigido. Anteriormente havia feito testes com valores muito pequenos.
    Primeira implementação com log de todas as métricas intermediárias para cálculo do logp da distribuição alvo para debug.
    """
    mlflow_client.create_experiment(
        experiment_name,
        tags={"mlflow.note.content": experiment_description}
    )

    neuron_options = [
        [16, 16],
        [16, 16, 16, 16],
        [64, 64],
        [64, 64, 64, 64],
        [256, 256]
    ]
    activation_options = ['gelu', 'tanh']
    max_epochs = 50
    learning_rate = 1e-3
    accelerator = 'cpu'

    for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
        job = MoonsTimeEmbeddingMLPCNFJob(
            name=f'exact-cnf-moons-{i}',
            dataset=dataset,
            model=TimeEmbeddingMLPCNFModel(
                vector_field=dict(
                    input_dims=2,
                    time_embed_dims=16,
                    time_embed_freq=100,
                    neurons_per_layer=neurons_per_layer + [2],
                    activations=(len(neurons_per_layer) + 1)*[activation],
                ),
                adjoint=True,
                base_distribution='standard_normal',
                max_epochs=max_epochs,
                checkpoint=dict(
                    monitor='val_loss',
                    mode='min',
                ),
                early_stopping=dict(
                    monitor='val_loss',
                    mode='min',
                    patience=3,
                    min_delta=1e-2,
                    stopping_threshold=-10
                ),
                input_shape=(2,),
                accelerator=accelerator,
                enable_progress_bar=False,
                learning_rate=learning_rate,
                max_time=dict(
                    hours=11
                )
            )
        )
        jobs_to_run.append((job, experiment_name))

    logs_dir = Path(__file__).parent / 'logs' / \
        f'run_moons_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    if debug:
        executor = submitit.DebugExecutor(folder=logs_dir)
    else:
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
        for job, mlflow_experiment_name in jobs_to_run:
            logger.info(f'Submitting job: {job.name}')
            executor.submit(run_job, job, mlflow_experiment_name)


if __name__ == "__main__":
    app()
