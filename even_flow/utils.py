import torch


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
