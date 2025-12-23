from typing import Any, ClassVar, Type

from .dataset import RingsDataset
from ..pydantic import MLFlowLoggedModel, YamlBaseModel
from ..jobs import BaseJob, DEFAULT_TRAINING_JOB_METRICS


class BaseRingGeneration(BaseJob, YamlBaseModel):

    MODEL_PREFIX: ClassVar[str] = 'model'
    DATASET_PREFIX: ClassVar[str] = 'dataset'
    METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.json'

    dataset: RingsDataset = RingsDataset()
    model: MLFlowLoggedModel
    metrics: dict[str, dict[str, float | int]
                  ] = DEFAULT_TRAINING_JOB_METRICS.copy()
