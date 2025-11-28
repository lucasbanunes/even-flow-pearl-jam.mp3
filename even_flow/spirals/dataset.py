from typing import Any, Self, Annotated
import sys
from random import randint
import lightning as L
import torch
from pydantic import Field

from ..utils import get_logger

type RandomState = int | None

# Annotated types with Pydantic Field validation
type InitialStateLowType = Annotated[
    int,
    Field(
        description="Lower bound for initial state generation."
    )
]

type InitialStateHighType = Annotated[
    int,
    Field(
        description="Upper bound for initial state generation."
    )
]

type NSeriesType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of spiral series to generate."
    )
]

type NTimestampsType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of time steps in each series."
    )
]

type TStepType = Annotated[
    float,
    Field(
        gt=0.0,
        description="Time step size for series generation."
    )
]

type DecayType = Annotated[
    float,
    Field(
        description="Exponential decay rate (should be negative for stable spirals)."
    )
]

type FrequencyType = Annotated[
    float,
    Field(
        gt=0.0,
        description="Frequency for sinusoidal components."
    )
]

type NoiseType = Annotated[
    tuple[float, float] | None,
    Field(
        description="Noise parameters as (mean, std) tuple, or None for no noise."
    )
]

type SamplesType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of samples for dataset split."
    )
]

type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for data loaders."
    )
]


def generate_series(
    initial_state_low: InitialStateLowType = 0,
    initial_state_high: InitialStateHighType = 1,
    n_series: NSeriesType = 100,
    n_timestamps: NTimestampsType = 1000,
    t_step: TStepType = 0.1,
    x_decay: DecayType = -0.1,
    y_decay: DecayType = -0.1,
    x_freq: FrequencyType = 1.0,
    y_freq: FrequencyType = 1.0,
    random_state: RandomState = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    if random_state is not None:
        generator.manual_seed(random_state)

    logger = get_logger()
    if x_decay >= 0 or y_decay >= 0:
        logger.warning("Exponential decay should be negative for stable spirals.")

    time = torch.arange(0, n_timestamps * t_step, t_step)
    x_radius_decay = torch.exp(x_decay * time)
    y_radius_decay = torch.exp(y_decay * time)
    cos = torch.cos(x_freq * time)
    sin = torch.sin(y_freq * time)
    all_states = []
    for _ in range(n_series):
        states = (initial_state_high - initial_state_low) * torch.rand((n_timestamps, 2), generator=generator) + initial_state_low
        states[:, 0] = states[:, 0] * x_radius_decay * cos
        states[:, 1] = states[:, 1] * y_radius_decay * sin
        all_states.append(states)

    all_states = torch.stack(all_states)
    time = torch.stack([time] * n_series)
    return time, all_states


class SpiralsDataModule(L.LightningDataModule):

    def __init__(self,
                 initial_state_low: InitialStateLowType = 0,
                 initial_state_high: InitialStateHighType = 1,
                 n_timestamps: NTimestampsType = 1000,
                 t_step: TStepType = 0.1,
                 x_decay: DecayType = -0.1,
                 y_decay: DecayType = -0.1,
                 x_freq: FrequencyType = 1.0,
                 y_freq: FrequencyType = 1.0,
                 x_noise: NoiseType = None,
                 y_noise: NoiseType = None,
                 train_samples: SamplesType = 1000,
                 val_samples: SamplesType = 200,
                 test_samples: SamplesType = 200,
                 batch_size: BatchSizeType = 32,
                 random_state: RandomState = None,):
        super(SpiralsDataModule, self).__init__()

        self.initial_state_low = initial_state_low
        self.initial_state_high = initial_state_high
        self.n_timestamps = n_timestamps
        self.t_step = t_step
        self.x_decay = x_decay
        self.y_decay = y_decay
        self.x_freq = x_freq
        self.y_freq = y_freq
        self.x_noise = x_noise
        self.y_noise = y_noise
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        if random_state is None:
            self.random_state = randint(-sys.maxsize, sys.maxsize)
        else:
            self.random_state = random_state

        self._train_series = None
        self._train_dataset = None
        self._train_dataloader = None

        self._val_series = None
        self._val_dataset = None
        self._val_dataloader = None

        self._test_series = None
        self._test_dataset = None
        self._test_dataloader = None

    def generate_dataloader(
            self,
            n_samples: int,
            random_state: RandomState = None,
            ) -> tuple[torch.Tensor, torch.utils.data.TensorDataset, torch.utils.data.DataLoader]:
        series = generate_series(
            initial_state_low=self.initial_state_low,
            initial_state_high=self.initial_state_high,
            n_timestamps=self.n_timestamps,
            t_step=self.t_step,
            x_decay=self.x_decay,
            y_decay=self.y_decay,
            x_freq=self.x_freq,
            y_freq=self.y_freq,
            n_series=n_samples,
            random_state=random_state,
        )
        if self.x_noise:
            series[1][:, 0] += torch.normal(self.x_noise[0], self.x_noise[1], size=series[1][:, 0].shape)
        if self.y_noise:
            series[1][:, 1] += torch.normal(self.y_noise[0], self.y_noise[1], size=series[1][:, 1].shape)
        dataset = torch.utils.data.TensorDataset(*series)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        return series, dataset, dataloader

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self._train_dataloader is None:
            self._train_series, self._train_dataset, self._train_dataloader = self.generate_dataloader(self.train_samples, self.random_state)
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self._val_dataloader is None:
            self._val_series, self._val_dataset, self._val_dataloader = self.generate_dataloader(self.val_samples, self.random_state+1)
        return self._val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self._test_dataloader is None:
            self._test_series, self._test_dataset, self._test_dataloader = self.generate_dataloader(self.test_samples, self.random_state+2)
        return self._test_dataloader

    @classmethod
    def pydantic_before_validator(cls, v: Any) -> Self:
        if isinstance(v, cls):
            return v
        elif v is None:
            return cls()
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Cannot convert {type(v)} to {cls}.")

    @staticmethod
    def pydantic_plain_serializer(v: 'SpiralsDataModule') -> dict[str, Any]:
        return {
            "initial_state_low": v.initial_state_low,
            "initial_state_high": v.initial_state_high,
            "n_timestamps": v.n_timestamps,
            "t_step": v.t_step,
            "x_decay": v.x_decay,
            "y_decay": v.y_decay,
            "x_freq": v.x_freq,
            "y_freq": v.y_freq,
            "x_noise": v.x_noise,
            "y_noise": v.y_noise,
            "train_samples": v.train_samples,
            "val_samples": v.val_samples,
            "test_samples": v.test_samples,
            "batch_size": v.batch_size,
            "random_state": v.random_state,
        }
