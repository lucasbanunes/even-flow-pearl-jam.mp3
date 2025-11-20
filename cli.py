import typer
import logging
import torch


from even_flow.utils import set_logger


set_logger()
logging.info(f'Is CUDA available? {torch.cuda.is_available()}')

app = typer.Typer(
    help="Even Flow CLI: tools for training and evaluating flow-based generative models."
)


if __name__ == "__main__":
    app()
