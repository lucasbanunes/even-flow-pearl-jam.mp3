from pathlib import Path
from typing import Annotated
import typer

from .jobs import (
    MoonsTimeEmbeddinngMLPNeuralODEJob,
    MoonsCNF
)

app = typer.Typer(
    help="Jobs for fitting a classifier on the Moons dataset."
)

type ConfigType = Annotated[
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
def cnf(
    config: ConfigType
) -> MoonsCNF:
    """Run a Moons CNF training job."""
    raise NotImplementedError("Moons CNF job is not yet implemented.")
