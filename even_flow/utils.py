from typing import Any
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


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary with dot-separated keys.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, optional
        The parent key prefix for nested keys, by default "".
    sep : str, optional
        The separator to use between keys, by default ".".

    Returns
    -------
    dict
        A flattened dictionary with dot-separated keys.

    Examples
    --------
    >>> nested_dict = {
    ...     'a': 2,
    ...     'b': {
    ...         'prop1': 1,
    ...         'prop2': 2
    ...     },
    ...     'c': ['shrebbles']
    ... }
    >>> flatten_dict(nested_dict)
    {'a': 2, 'b.prop1': 1, 'b.prop2': 2, 'c.0': 'shrebbles'}
    """
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(value, new_key, sep).items())
        elif isinstance(value, list):
            # Handle lists by using index as key
            for i, item in enumerate(value):
                list_key = f"{new_key}{sep}{i}"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep).items())
                else:
                    items.append((list_key, item))
        else:
            # For primitive values, add directly
            items.append((new_key, value))

    return dict(items)
