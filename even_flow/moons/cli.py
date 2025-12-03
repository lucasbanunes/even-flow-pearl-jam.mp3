from pathlib import Path
from typing import Annotated
import typer

from .jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsTimeEmbeddingMLPCNFJob,
    MoonsTimeEmbeddingMLPCNFHutchinsonJob,
    MoonsRealNVPJob,
    MoonsZukoCNFJob
)

app = typer.Typer(
    help="Jobs for fitting a classifier on the Moons dataset."
)

ConfigType = Annotated[
    Path,
    typer.Option('--config',
                 help="Path to the yaml configuration file for the job.")
]


@app.command()
def time_embedding_mlp_neural_ode(
    config: ConfigType
) -> MoonsTimeEmbeddinngMLPNeuralODEJob:
    """Run a Moons Time Embedding MLP Neural ODE training job."""
    job = MoonsTimeEmbeddinngMLPNeuralODEJob.from_yaml(
        config)
    job.run()
    return job


@app.command()
def time_embedding_mlp_cnf(
    config: ConfigType
) -> MoonsTimeEmbeddingMLPCNFJob:
    """Run a Moons Time Embedding MLP CNF training job."""
    job = MoonsTimeEmbeddingMLPCNFJob.from_yaml(config)
    job.run()
    return job


@app.command()
def time_embedding_mlp_cnf_hutchingson(
    config: ConfigType
) -> MoonsTimeEmbeddingMLPCNFHutchinsonJob:
    """Run a Moons Time Embedding MLP CNF Hutchingson training job."""
    job = MoonsTimeEmbeddingMLPCNFHutchinsonJob.from_yaml(config)
    job.run()
    return job


@app.command()
def real_nvp(
    config: ConfigType
) -> MoonsRealNVPJob:
    """Run a Moons Real NVP training job."""
    job = MoonsRealNVPJob.from_yaml(config)
    job.run()
    return job


@app.command()
def zuko_cnf(
    config: ConfigType
) -> MoonsZukoCNFJob:
    """Run a Moons Zuko CNF training job."""
    job = MoonsZukoCNFJob.from_yaml(config)
    job.run()
    return job
