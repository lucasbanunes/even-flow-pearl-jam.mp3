import torch
import logging
import logging.config

EVENFLOW_LOGGER_NAME = "even_flow"


def interleave_columns(*args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Interleave multiple tensors column-wise.

    Parameters
    ----------
    *args : tuple of torch.Tensor
        Tensors to interleave. All tensors must have the same number of rows.

    Returns
    -------
    torch.Tensor
        A single tensor with columns interleaved from the input tensors.
    """
    return torch.flatten(torch.stack(args, dim=2), start_dim=1)


def set_logger(level="INFO", name="") -> logging.Logger:
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(module)s"
                " | %(lineno)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
        },
        "loggers": {
            name: {
                "level": level,
                "handlers": ["stdout"]
            }
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger(name)


def get_logger() -> logging.Logger:
    return logging.getLogger(EVENFLOW_LOGGER_NAME)
