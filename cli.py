import typer
import torch
import mplhep
import matplotlib.pyplot as plt

from even_flow.utils import get_logger, set_logger
from even_flow.moons.cli import app as moons_app

plt.style.use(mplhep.style.ATLAS)
set_logger()
logger = get_logger()
logger.info(f'Is CUDA available? {torch.cuda.is_available()}')

app = typer.Typer(
    help="Even Flow CLI: tools for training and evaluating flow-based generative models."
)
app.add_typer(moons_app, name="moons")


if __name__ == "__main__":
    app()
