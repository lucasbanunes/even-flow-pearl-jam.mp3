import typer
import torch
import mplhep
import matplotlib.pyplot as plt

from even_flow.utils import get_logger, set_logger
from even_flow.moons.cli import app as moons_app
from even_flow.mnist.cli import app as mnist_app

plt.style.use(mplhep.style.ATLAS)
set_logger()
logger = get_logger()
logger.info(f'Is CUDA available? {torch.cuda.is_available()}')

app = typer.Typer(
    help="Even Flow CLI: tools for training and evaluating flow-based generative models."
)
app.add_typer(moons_app, name="moons")
app.add_typer(mnist_app, name="mnist")


if __name__ == "__main__":
    app()
