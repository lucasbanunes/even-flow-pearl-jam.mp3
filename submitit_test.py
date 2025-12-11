import submitit
import time
from pathlib import Path
import shutil


def my_cool_function(value):
    print('Received:', value)
    time.sleep(10)
    return value


logs_dir = Path.home() / 'logs' / 'submitit_logs'
if logs_dir.exists():
    shutil.rmtree(str(logs_dir))
logs_dir.mkdir(parents=True, exist_ok=True)
executor = submitit.AutoExecutor(folder=logs_dir)
executor.update_parameters(slurm_array_parallelism=2,
                           timeout_min=10,
                           slurm_partition="gpu")
jobs = []
with executor.batch():
    for k in range(1, 11):
        job = executor.submit(my_cool_function, k)
        jobs.append(job)
# wait and check how many have finished
time.sleep(11)
num_finished = sum(job.done() for job in jobs)
print(num_finished)  # probably around 2 have finished, given the overhead

# then you may want to wait until all jobs are completed:
outputs = [job.result() for job in jobs]
print(outputs)
