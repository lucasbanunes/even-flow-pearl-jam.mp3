from even_flow.utils import get_logger
from even_flow.moons.jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsRealNVPJob,
    MoonsTimeEmbeddingMLPCNFJob,
    MoonsTimeEmbeddingMLPCNFHutchinsonJob
)
from even_flow.moons.dataset import MoonsDataset
from even_flow.models.neuralode import TimeEmbeddingMLPNeuralODEModel
from even_flow.models.real_nvp import RealNVPModel
from even_flow.models.cnf import TimeEmbeddingMLPCNFModel
from itertools import product
import mlflow
import warnings
import numpy as np
warnings.filterwarnings("ignore")


logger = get_logger()

train_samples = 10000
val_samples = 1000
test_samples = 1000
noise = 0.05
batch_size = 32
random_state = 943874

datamodule = MoonsDataset(
    train_samples=train_samples,
    val_samples=val_samples,
    test_samples=test_samples,
    noise=noise,
    batch_size=batch_size,
    random_state=random_state
)

logger.info('Running MoonsTimeEmbeddinngMLPNeuralODEJob...')

rtols = np.logspace(-2, -7, 10)
atols = np.logspace(-2, -7, 10)
solvers = ['euler', 'dopri5', 'rk4']
neurons = [[16, 2], [16, 16, 2], [16, 16, 16, 2]]
neural_ode_jobs = [
    MoonsTimeEmbeddinngMLPNeuralODEJob(
        datamodule=datamodule,
        name=f'time-embedding-mlp-neural-ode-{i}',
        max_epochs=3,
        model=TimeEmbeddingMLPNeuralODEModel(
            input_shape=(2,),
            vector_field=dict(
                input_dims=2,
                time_embed_dims=2,
                time_embed_freq=10,
                neurons_per_layer=neuron_layers,
                activations=['relu']*len(neuron_layers),
            ),
            atol=atol,
            rtol=rtol,
            solver=solver,
            max_epochs=100,
            monitor='val_loss',
            mode='min',
            min_delta=1e-3,
            patience=5,
            verbose=False,
        ),

    ) for i, (rtol, atol, solver, neuron_layers) in enumerate(
        product(rtols, atols, solvers, neurons))
]
logger.info(f'Running {len(neural_ode_jobs)} jobs.')

mlflow.set_experiment("Moons Neural ODE")

for job in neural_ode_jobs:
    logger.info(
        f'Runnning job: {job.name} | Solver: {job.model.solver} | rtol: {job.model.rtol} | atol: {job.model.atol}')
    job.run()

mlflow.set_experiment('Moons Real NVP')

hidden_features_options = [
    [64, 64],
    [64, 64, 64, 64],
    [256, 256]
]

realnvp_jobs = [
    MoonsRealNVPJob(
        name=f'real-nvp-moons-{i}',
        datamodule=datamodule,
        model=RealNVPModel(
            features=2,
            transforms=4,
            hidden_features=hidden_feature,
            monitor='val_loss',
            mode='min',
            max_epochs=50,
        )
    )
    for i, hidden_feature in enumerate(hidden_features_options)
]
for job in realnvp_jobs:
    logger.info(
        f'Running job: {job.name} | Hidden features: {job.model.hidden_features}')
    job.run()


mlflow.set_experiment('Moons Exact CNF')

neurons = hidden_features_options.copy()
activation_options = ['relu', 'gelu']

exact_cnf_jobs = [
    MoonsTimeEmbeddingMLPCNFJob(
        name=f'exact-cnf-moons-{i}',
        datamodule=datamodule,
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
            max_epochs=50,
            patience=3,
            verbose=True,
            min_delta=1,
            input_shape=(2,),
            monitor='val_loss',
            mode='min',
        )
    )
    for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neurons))
]
for job in exact_cnf_jobs:
    logger.info(f'Running job: {job.name}')
    job.run()


mlflow.set_experiment('Moons Hutchinson CNF')

neurons = hidden_features_options.copy()
activation_options = ['relu', 'gelu']

hutchinson_cnf_jobs = [
    MoonsTimeEmbeddingMLPCNFHutchinsonJob(
        name=f'hutchinson-cnf-moons-{i}',
        datamodule=datamodule,
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
            max_epochs=50,
            patience=3,
            verbose=True,
            min_delta=1,
            input_shape=(2,),
            monitor='val_loss',
            mode='min',
        )
    )
    for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neurons))
]
for job in hutchinson_cnf_jobs:
    logger.info(f'Running job: {job.name}')
    job.run()
