import os
import mlflow
from typing import Any

from .pydantic import MLFlowLoggedModel


class SlurmEnvironment(MLFlowLoggedModel):
    """Pydantic model representing a SLURM environment."""
    job_id: str | None = os.getenv('SLURM_JOB_ID', None)
    job_name: str | None = os.getenv('SLURM_JOB_NAME', None)
    array_job_id: str | None = os.getenv('SLURM_ARRAY_JOB_ID', None)
    array_task_id: str | None = os.getenv('SLURM_ARRAY_TASK_ID', None)

    def _to_mlflow(self, prefix=''):
        """Log SLURM environment variables to MLflow."""
        mlflow.log_param(f'{prefix}slurm_job_id', self.job_id)
        mlflow.log_param(f'{prefix}slurm_job_name', self.job_name)
        mlflow.log_param(f'{prefix}slurm_array_job_id', self.array_job_id)
        mlflow.log_param(f'{prefix}slurm_array_task_id', self.array_task_id)

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs) -> dict[str, Any]:
        """Load SLURM environment variables from MLflow."""
        job_id = mlflow_run.data.params.get(f'{prefix}slurm_job_id', None)
        job_name = mlflow_run.data.params.get(f'{prefix}slurm_job_name', None)
        array_job_id = mlflow_run.data.params.get(
            f'{prefix}slurm_array_job_id', None)
        array_task_id = mlflow_run.data.params.get(
            f'{prefix}slurm_array_task_id', None)

        kwargs['job_id'] = job_id if job_id != 'None' else None
        kwargs['job_name'] = job_name if job_name != 'None' else None
        kwargs['array_job_id'] = array_job_id if array_job_id != 'None' else None
        kwargs['array_task_id'] = array_task_id if array_task_id != 'None' else None

        return kwargs
