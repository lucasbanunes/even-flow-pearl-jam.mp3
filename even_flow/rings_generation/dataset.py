from pathlib import Path
from typing import Annotated, Any
from pydantic import Field
import mlflow
import lightning as L

from ..pydantic import MLFlowLoggedModel


type RingsColType = Annotated[
    str,
    Field(
        description="Name of the column containing ring labels."
    )
]


type FileType = Annotated[
    Path,
    Field(
        description="Path to the file containing the rings data."
    )
]

type RandomState = Annotated[
    int | None,
    Field(
        description="Random state for reproducibility."
    )
]


type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for data loaders."
    )
]


class RingsDataModule(L.LightningDataModule):
    def __init__(self,
                 rings_col: RingsColType,
                 file: FileType,
                 batch_size: BatchSizeType = 32,
                 random_state: RandomState = 42):
        super().__init__()
        self.rings_col = rings_col
        self.file = file
        self.batch_size = batch_size
        self.random_state = random_state


class RingsDataset(MLFlowLoggedModel):

    rings_col: RingsColType
    file: FileType
    batch_size: BatchSizeType = 32
    random_state: RandomState = 42

    def _to_mlflow(self, prefix: str = ''):
        mlflow.log_param(f'{prefix}.rings_col', self.rings_col)
        mlflow.log_param(f'{prefix}.file', str(self.file))
        mlflow.log_param(f'{prefix}.batch_size', self.batch_size)
        mlflow.log_param(f'{prefix}.random_state', self.random_state)

    @classmethod
    def _from_mlflow(cls, mlflow_run: mlflow.entities.Run, prefix: str = '', **kwargs) -> dict[str, Any]:
        if prefix:
            prefix += '.'
        kwargs['rings_col'] = mlflow_run.data.params.get(
            f'{prefix}rings_col', cls.model_fields['rings_col'].default)
        kwargs['file'] = Path(mlflow_run.data.params.get(
            f'{prefix}file', cls.model_fields['file'].default))
        kwargs['batch_size'] = int(mlflow_run.data.params.get(
            f'{prefix}batch_size', cls.model_fields['batch_size'].default))
        random_state_param = mlflow_run.data.params.get(
            f'{prefix}random_state', cls.model_fields['random_state'].default)
        kwargs['random_state'] = int(
            random_state_param) if random_state_param is not None else None
        return kwargs
