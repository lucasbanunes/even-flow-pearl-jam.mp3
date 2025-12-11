from even_flow.utils import set_logger
from even_flow.moons.jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsRealNVPJob,
    MoonsZukoCNFJob,
    MoonsTimeEmbeddingMLPCNFJob,
    MoonsTimeEmbeddingMLPCNFHutchinsonJob,
    MoonsTimeEmbeddingMLPCNFTorchJob
)
from even_flow.moons.dataset import MoonsDataset
from even_flow.models.neuralode import TimeEmbeddingMLPNeuralODEModel
from even_flow.models.real_nvp import RealNVPModel
from even_flow.models.cnf import (
    TimeEmbeddingMLPCNFModel,
    TimeEmbeddingMLPCNFHutchinsonModel,
    ZukoCNFModel,
    TimeEmbeddingMLPCNFTorchModel
)
from itertools import product
import mlflow
import warnings
import numpy as np
import submitit
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")


logger = set_logger()

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

experiment_name = "Moons Neural ODE"
rtols = np.logspace(-2, -7, 5)
atols = np.logspace(-2, -7, 5)
solvers = ['euler', 'dopri5', 'rk4']
neurons = [[16, 2], [16, 16, 2], [16, 16, 16, 2]]
max_epochs = 50
learning_rate = 1e-3
accelerator = 'cpu'

jobs_to_run = []


def run_job(job, mlflow_experiment_name):
    mlflow.set_experiment(mlflow_experiment_name)
    job.run()


for i, (rtol, atol, solver, neuron_layers) in enumerate(product(rtols, atols, solvers, neurons)):
    logger.info(
        f'Runnning job: time-embedding-mlp-neural-ode-{i} | Solver: {solver} | rtol: {rtol} | atol: {atol}')
    if solver in ['euler', 'rk4']:
        ode_options = dict(step_size=0.1)
    else:
        ode_options = dict()
    job = MoonsTimeEmbeddinngMLPNeuralODEJob(
        dataset=dataset,
        name=f'time-embedding-mlp-neural-ode-{i}',
        model=TimeEmbeddingMLPNeuralODEModel(
            input_shape=(2,),
            vector_field=dict(
                input_dims=2,
                time_embed_dims=2,
                time_embed_freq=10,
                neurons_per_layer=neuron_layers,
                activations=['gelu']*len(neuron_layers),
            ),
            atol=atol,
            rtol=rtol,
            solver=solver,
            max_epochs=max_epochs,
            early_stopping=dict(
                monitor='val_loss',
                mode='min',
                patience=5,
                min_delta=1e-2,
                stopping_threshold=-10
            ),
            checkpoint=dict(
                monitor='val_loss',
                mode='min',
            ),
            learning_rate=1e-3,
            enable_progress_bar=False,
            accelerator=accelerator,
            ode_options=ode_options
        ),
    )
    jobs_to_run.append((job, experiment_name))


neuron_options = [
    [16, 16],
    [16, 16, 16, 16],
    [64, 64],
    [64, 64, 64, 64],
    [256, 256]
]
activation_options = ['gelu', 'tanh']
experiment_name = 'Moons Torch CNF'


# for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
#     job = MoonsTimeEmbeddingMLPCNFTorchJob(
#         name=f'torch-cnf-moons-{i}',
#         dataset=dataset,
#         model=TimeEmbeddingMLPCNFTorchModel(
#             vector_field=dict(
#                 input_dims=2,
#                 time_embed_dims=16,
#                 time_embed_freq=100,
#                 neurons_per_layer=neurons + [2],
#                 activations=(len(neurons) + 1)*[activation],
#             ),
#             adjoint=False,
#             base_distribution='standard_normal',
#             max_epochs=max_epochs,
#             input_shape=(2,),
#         )
#     )
#     job.run()


experiment_name = 'Moons Zuko Hutchinson CNF'


# for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
#     job = MoonsZukoCNFJob(
#         name=f'zuko-hutchinson-cnf-moons-{i}',
#         dataset=dataset,
#         model=ZukoCNFModel(
#             features=2,
#             hidden_features=neurons,
#             activation=activation,
#             exact=False,
#             max_epochs=max_epochs,
#             checkpoint=dict(
#                 monitor='val_loss',
#                 mode='min',
#             ),
#             early_stopping=dict(
#                 monitor='val_loss',
#                 mode='min',
#                 patience=3,
#                 min_delta=1e-2,
#                 stopping_threshold=-10
#             ),
#             learning_rate=learning_rate,
#             accelerator=accelerator,
#             enable_progress_bar=False
#         )
#     )
#     jobs_to_run.append((job, experiment_name))


experiment_name = 'Moons Real NVP'


# for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
#     job = MoonsRealNVPJob(
#         name=f'real-nvp-moons-{i}',
#         dataset=dataset,
#         model=RealNVPModel(
#             features=2,
#             transforms=4,
#             hidden_features=neurons,
#             checkpoint=dict(
#                 monitor='val_loss',
#                 mode='min',
#             ),
#             early_stopping=dict(
#                 monitor='val_loss',
#                 mode='min',
#                 patience=5,
#                 min_delta=1e-2,
#                 stopping_threshold=-10
#             ),
#             learning_rate=learning_rate,
#             max_epochs=max_epochs,
#             activation=activation,
#             accelerator=accelerator,
#             enable_progress_bar=False
#         )
#     )
#     jobs_to_run.append((job, experiment_name))


experiment_name = 'Moons Zuko Exact CNF'


# for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
#     job = MoonsZukoCNFJob(
#         name=f'zuko-exact-cnf-moons-{i}',
#         dataset=dataset,
#         model=ZukoCNFModel(
#             features=2,
#             hidden_features=neurons,
#             activation=activation,
#             exact=True,
#             max_epochs=max_epochs,
#             checkpoint=dict(
#                 monitor='val_loss',
#                 mode='min',
#             ),
#             early_stopping=dict(
#                 monitor='val_loss',
#                 mode='min',
#                 patience=3,
#                 min_delta=1e-2,
#                 stopping_threshold=-10
#             ),
#             learning_rate=learning_rate,
#             accelerator=accelerator,
#             enable_progress_bar=False
#         )
#     )
#     jobs_to_run.append((job, experiment_name))


# mlflow.set_experiment('Moons Exact CNF')

# neuron_options = [
#     [2, 2],
#     [2, 2, 2, 2],
#     [8, 8]
# ]
# for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
#     job = MoonsTimeEmbeddingMLPCNFJob(
#         name=f'exact-cnf-moons-{i}',
#         dataset=dataset,
#         model=TimeEmbeddingMLPCNFModel(
#             vector_field=dict(
#                 input_dims=2,
#                 time_embed_dims=16,
#                 time_embed_freq=100,
#                 neurons_per_layer=neurons_per_layer + [2],
#                 activations=(len(neurons_per_layer) + 1)*[activation],
#             ),
#             adjoint=True,
#             base_distribution='standard_normal',
#             max_epochs=5,
#             patience=1,
#             min_delta=1,
#             input_shape=(2,),
#             monitor='val_loss',
#             mode='min',
#         )
#     )
#     job.run()


# mlflow.set_experiment('Moons Hutchinson CNF')

# for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
#     job = MoonsTimeEmbeddingMLPCNFHutchinsonJob(
#         name=f'hutchinson-cnf-moons-{i}',
#         dataset=dataset,
#         model=TimeEmbeddingMLPCNFHutchinsonModel(
#             vector_field=dict(
#                 input_dims=2,
#                 time_embed_dims=16,
#                 time_embed_freq=100,
#                 neurons_per_layer=neurons_per_layer + [2],
#                 activations=(len(neurons_per_layer) + 1)*[activation],
#             ),
#             adjoint=True,
#             base_distribution='standard_normal',
#             max_epochs=5,
#             patience=1,
#             min_delta=1,
#             monitor='val_loss',
#             mode='min',
#             input_shape=(2,),
#         )
#     )
#     job.run()


logs_dir = Path.home() / 'logs' / \
    f'run_moons_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
executor = submitit.AutoExecutor(folder=logs_dir)
executor.update_parameters(slurm_array_parallelism=2,
                           timeout_min=3*60,
                           cpus_per_task=8,
                           slurm_partition="gpu")
with executor.batch():
    for job, mlflow_experiment_name in jobs_to_run:
        logger.info(f'Submitting job: {job.name}')
        executor.submit(run_job, job, mlflow_experiment_name)
