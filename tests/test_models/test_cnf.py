import torch
from even_flow.models.cnf import CNF1D


def test_cnf1d_initialization():
    """Test the initialization of the CNF1D model."""
    vector_field = torch.nn.Linear(2, 2)
    cnf_model = CNF1D(
        vector_field=vector_field,
        adjoint=True,
        solver='dopri5',
        atol=1e-5,
        rtol=1e-5,
        learning_rate=1e-3
    )
    assert cnf_model.vector_field == vector_field
    assert cnf_model.adjoint is True
    assert cnf_model.solver == 'dopri5'
    assert cnf_model.atol == 1e-5
    assert cnf_model.rtol == 1e-5
    assert cnf_model.learning_rate == 1e-3

    x = torch.ones((2, 2), dtype=torch.float32)
    cnf_model.forward(x)  # Just to check that forward runs without error
