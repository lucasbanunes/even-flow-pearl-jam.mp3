from pathlib import Path
from typing import Annotated
import typer

from .jobs import MNISTZukoCNFJob, MNISTRealNVPJob


app = typer.Typer(
    help="Jobs for fitting a classifier on the MNIST dataset."
)


ConfigType = Annotated[
    Path,
    typer.Option('--config',
                 help="Path to the yaml configuration file for the job.")
]


@app.command()
def zuko_cnf(
    config: ConfigType
) -> MNISTZukoCNFJob:
    """Run a MNIST Zuko CNF training job."""
    job = MNISTZukoCNFJob.from_yaml(config)
    job.run()
    return job


@app.command()
def real_nvp(
    config: ConfigType
) -> MNISTRealNVPJob:
    """Run a MNIST Real NVP training job."""
    job = MNISTRealNVPJob.from_yaml(config)
    job.run()
    return job
