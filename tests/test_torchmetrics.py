from torchmetrics import MeanMetric, MetricCollection
import torch
import logging


def test_multiple_means():

    collection = MetricCollection({
        'x_mean': MeanMetric(),
        'y_mean': MeanMetric()
    })

    # 2. Simulate data
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # 3. Update by passing kwargs that match the keys in the collection
    # 'x_mean' receives variable x, 'y_mean' receives variable y
    for i in range(len(x)):
        collection['x_mean'].update(x[i])
        collection['y_mean'].update(y[i])

    # 4. Compute results
    results = collection.compute()
    logging.info(f'Computed metrics: {results}')
    logging.info(f'Expected x mean: {x.mean()}, y mean: {y.mean()}')
    assert torch.isclose(results['x_mean'], x.mean())
    assert torch.isclose(results['y_mean'], y.mean())
