from pathlib import Path
from typing import Annotated
import typer

from .jobs import MoonsTimeEmbeddinngMLPNeuralODE

app = typer.Typer(
    help="Jobs for fitting a classifier on the Moons dataset."
)


@app.command()
def time_embedding_neural_ode(
    config: Annotated[
        Path,
        typer.Option('--config',
                     help="Path to the yaml configuration file for the job.")
    ]
) -> MoonsTimeEmbeddinngMLPNeuralODE:
    """Run a Moons Time Embedding MLP Neural ODE training job."""
    job = MoonsTimeEmbeddinngMLPNeuralODE.from_yaml(
        config)
    job.run()
    return job
