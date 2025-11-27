import torch
from torch.func import jacrev, jvp, vmap

x = torch.Tensor([[1.0, 2.0], [1.0, 2.0]]).to(
    torch.float32)  # (batch_size, dim)
print(f'x:\n{x}')


def vector_field(x):
    # Example: y[0] = x[0]^3, y[1] = x[1]^2
    return torch.stack([x[0]**3, x[1]**2])


vector_field_vectorized = vmap(vector_field)
vetor_field_value = vector_field_vectorized(x)
print(f'Vector field value:\n{vetor_field_value}')


# def vector_field_jacobian(x):
#     return torch.Tensor([
#         [3*x[0]**2, 0],
#         [0, 2*x[1]]
#     ])


# vector_field_jacobian = vmap(vector_field_jacobian)
torch_vector_field_jacobian = vmap(jacrev(vector_field))
# explicit_jacobian = vector_field_jacobian(x)
torch_jacobian = torch_vector_field_jacobian(x)
torch_jacobian_div = torch.diagonal(torch_jacobian, dim1=1, dim2=2).sum(-1)

# print(f'Explicit Jacobian:\n{explicit_jacobian}')
print(f'Torch Jacobian:\n{torch_jacobian}')
print(f'Torch Jacobian divergence:\n{torch_jacobian_div}')
# jacobians_equal = torch.allclose(explicit_jacobian, torch_jacobian)
# print(f'Jacobians equal: {jacobians_equal}')


def vector_field_divergence(x):
    div = 3*x[0]**2 + 2*x[1]
    return div


vector_field_divergence = vmap(vector_field_divergence)


def torch_vector_field_divergence(x) -> tuple[torch.Tensor, torch.Tensor]:
    divergence = 0
    for i in range(len(x)):
        e_i = torch.zeros_like(x)
        e_i[i] = 1.0
        x_eval, jvp_result = jvp(vector_field, (x,), (e_i,))
        print(jvp_result)
        divergence += jvp_result.flatten()[i]
    return x_eval, divergence


torch_vector_field_divergence = vmap(torch_vector_field_divergence)


explicit_divergence = vector_field_divergence(x)
x_eval, torch_divergence = torch_vector_field_divergence(x)
print(f'Explicit divergence: {explicit_divergence}')
print(f'Torch divergence: {torch_divergence}')
print(f'Torch divergence x eval: {x_eval}')
divergences_equal = torch.allclose(explicit_divergence, torch_divergence)
print(f'Divergences equal: {divergences_equal}')
