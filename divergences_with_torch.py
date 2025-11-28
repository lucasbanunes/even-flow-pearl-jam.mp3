import torch

x = torch.Tensor([[1.0, 2.0], [3, 1.0]]).to(
    torch.float32)  # (batch_size, dim)
t = torch.zeros(x.shape[0], dtype=x.dtype)  # (batch_size,)
print(f'x:\n{x}')


def vector_field(t, x):
    # Example: y[0] = x[0]^3, y[1] = x[1]^2
    return torch.stack([x[:, 0]**3, x[:, 1]**2], dim=-1)


vector_field_value = vector_field(t, x)
print(f'Vector field value:\n{vector_field_value}')


def vector_field_divergence(t, x):
    div = 3*x[:, 0]**2 + 2*x[:, 1]
    return div``


def compute_exact_divergence(t, x):
    """
    Computes exact divergence using a loop over dimensions with standard autograd.
    Complexity: O(Dim) backprops.
    """
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        dx = vector_field(t, x)

        trace = torch.zeros(x.shape[0], dtype=x.dtype)

        grad_outputs = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        grad_outputs = grad_outputs.expand(*x.shape, -1).movedim(-1, 0)

        (jacobian,) = torch.autograd.grad(
            dx, x,
            grad_outputs=grad_outputs,
            create_graph=True, is_grads_batched=True
        )
        trace = torch.einsum("i...i", jacobian)

    return trace


explicit_divergence = vector_field_divergence(t, x)
torch_autograd_divergence = compute_exact_divergence(t, x)
print(f'Explicit divergence:\n{explicit_divergence}')
print(f'Torch autograd divergence:\n{torch_autograd_divergence}')
autograd_equal = torch.allclose(
    explicit_divergence, torch_autograd_divergence)
print(f'Torch autograd divergence equal: {autograd_equal}')
