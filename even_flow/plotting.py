
import matplotlib.pyplot as plt
import torch
import numpy as np


@torch.no_grad()
def quiver_plot(x, y, vector_field, t=0.0, ax=None,
                colorbar: bool = True,
                cmap='coolwarm'):
    if ax is None:
        _, ax = plt.subplots()
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        dx = vector_field(torch.Tensor([t]), grid_points)
    U = dx[:, 0].reshape(X.shape).detach().cpu().numpy()
    V = dx[:, 1].reshape(Y.shape).detach().cpu().numpy()
    M = np.hypot(U, V)
    q = ax.quiver(X, Y, U, V, M,
                  cmap=cmap)
    c = None
    if colorbar:
        c = plt.colorbar(q, ax=ax, label='Vector Magnitude')

    return X, Y, U, V, M, q, c
