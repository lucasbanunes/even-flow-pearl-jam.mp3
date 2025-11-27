from typing import Annotated
from pydantic import Field
from pathlib import Path


type CheckpointsDirType = Annotated[
    Path | None,
    Field(
        description="Directory to save model checkpoints."
    )
]


type MaxEpochsType = Annotated[
    int,
    Field(
        description="Maximum number of epochs for training."
    )
]


type PatienceType = Annotated[
    int,
    Field(
        description="Number of epochs with no improvement after which training will be stopped."
    )
]

type TrainerAcceleratorType = Annotated[
    str,
    Field(
        description="The accelerator to use for training (e.g., 'cpu', 'gpu')."
    )
]

type TrainerTestVerbosityType = Annotated[
    bool,
    Field(
        description="Whether to enable verbose logging during testing."
    )
]