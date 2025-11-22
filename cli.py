import typer
import logging
import torch


from even_flow.utils import set_logger
from even_flow.moons.cli import app as moons_app


set_logger()
logging.info(f'Is CUDA available? {torch.cuda.is_available()}')

app = typer.Typer(
    help="Even Flow CLI: tools for training and evaluating flow-based generative models."
)
app.add_typer(moons_app, name="moons")


if __name__ == "__main__":
    app()
