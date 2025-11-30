from even_flow.utils import set_logger
from even_flow.moons.jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsRealNVPJob,
    MoonsTimeEmbeddingMLPCNFJob,
    MoonsTimeEmbeddingMLPCNFHutchinsonJob
)
from even_flow.moons.dataset import MoonsDataset
from even_flow.models.neuralode import TimeEmbeddingMLPNeuralODEModel
from even_flow.models.real_nvp import RealNVPModel
from even_flow.models.cnf import (
    TimeEmbeddingMLPCNFModel,
    TimeEmbeddingMLPCNFHutchinsonModel
)
from itertools import product
import mlflow
import warnings
# import numpy as np
warnings.filterwarnings("ignore")


logger = set_logger()

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

# mlflow.set_experiment("Moons Neural ODE")
# rtols = np.logspace(-2, -7, 10)
# atols = np.logspace(-2, -7, 10)
# solvers = ['euler', 'dopri5', 'rk4']
# neurons = [[16, 2], [16, 16, 2], [16, 16, 16, 2]]
# for i, (rtol, atol, solver, neuron_layers) in enumerate(product(rtols, atols, solvers, neurons)):
#     logger.info(
#         f'Runnning job: time-embedding-mlp-neural-ode-{i} | Solver: {solver} | rtol: {rtol} | atol: {atol}')
#     job = MoonsTimeEmbeddinngMLPNeuralODEJob(
#         datamodule=datamodule,
#         name=f'time-embedding-mlp-neural-ode-{i}',
#         max_epochs=3,
#         model=TimeEmbeddingMLPNeuralODEModel(
#             input_shape=(2,),
#             vector_field=dict(
#                 input_dims=2,
#                 time_embed_dims=2,
#                 time_embed_freq=10,
#                 neurons_per_layer=neuron_layers,
#                 activations=['relu']*len(neuron_layers),
#             ),
#             atol=atol,
#             rtol=rtol,
#             solver=solver,
#             max_epochs=100,
#             monitor='val_loss',
#             mode='min',
#             min_delta=1e-3,
#             patience=5,
#             verbose=False,
#         ),
#     )
#     job.run()


mlflow.set_experiment('Moons Real NVP')

neuron_options = [
    [64, 64],
    [64, 64, 64, 64],
    [256, 256]
]
activation_options = ['gelu', 'tanh']
for i, (neurons, activation) in enumerate(product(neuron_options, activation_options)):
    logger.info(
        f'Running job: real-nvp-moons-{i} | Hidden features: {neurons} | Activation: {activation}')
    job = MoonsRealNVPJob(
        name=f'real-nvp-moons-{i}',
        datamodule=datamodule,
        model=RealNVPModel(
            features=2,
            transforms=4,
            hidden_features=neurons,
            monitor='val_loss',
            mode='min',
            patience=5,
            min_delta=1e-3,
            max_epochs=50,
            activation=activation
        )
    )
    job.run()


# mlflow.set_experiment('Moons Exact CNF')

# neuron_options = [
#     [2, 2],
#     [2, 2, 2, 2],
#     [8, 8]
# ]
# for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
#     logger.info(f'Running job: exact-cnf-moons-{i}')
#     job = MoonsTimeEmbeddingMLPCNFJob(
#         name=f'exact-cnf-moons-{i}',
#         datamodule=datamodule,
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
#             verbose=True,
#             min_delta=1,
#             input_shape=(2,),
#             monitor='val_loss',
#             mode='min',
#         )
#     )
#     job.run()


# mlflow.set_experiment('Moons Hutchinson CNF')

# for i, (activation, neurons_per_layer) in enumerate(product(activation_options, neuron_options)):
#     logger.info(f'Running job: hutchinson-cnf-moons-{i}')
#     job = MoonsTimeEmbeddingMLPCNFHutchinsonJob(
#         name=f'hutchinson-cnf-moons-{i}',
#         datamodule=datamodule,
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
#             verbose=True,
#             min_delta=1,
#             monitor='val_loss',
#             mode='min',
#             input_shape=(2,),
#         )
#     )
#     job.run()
