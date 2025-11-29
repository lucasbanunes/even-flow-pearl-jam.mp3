from even_flow.models.cnf import TimeEmbeddingMLPCNFModel


def test_cnf_sample():
    cnf_model = TimeEmbeddingMLPCNFModel(
        vector_field=dict(
            input_dims=2,
            time_embed_dims=3,
            time_embed_freq=10,
            neurons_per_layer=[13, 13, 2],
            activations=['tanh', 'tanh', 'linear'],
        ),
        solver='dopri5',
        atol=1e-5,
        rtol=1e-5,
        input_shape=(2,)
    )
    base, transformed = cnf_model.sample(shape=(10,))
    assert base.shape == (10, 2), "Sample shape is incorrect."
    assert transformed.shape == (10, 2), "Transformed shape is incorrect."
