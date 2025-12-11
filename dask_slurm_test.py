import joblib
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import time

cluster = SLURMCluster(
    queue="gpu",
    project="even_flow",
    cores=8,
    memory='16GB',
    walltime="02:00:00"
)

cluster.adapt(maximum_jobs=4)  # Request up to 4 jobs


dask_client = Client(cluster)

print('Dashboard link:', dask_client.dashboard_link)


def my_cool_function(value):
    print('Received:', value)
    time.sleep(120)
    return value


with joblib.parallel_backend('dask'):
    results = joblib.Parallel()(joblib.delayed(my_cool_function)(i)
                                for i in range(1, 11))

print(results)
