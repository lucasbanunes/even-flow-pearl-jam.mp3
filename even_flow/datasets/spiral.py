from typing_extensions import Annotated
import lightning as L
import torch
from pydantic import Field

from ..utils import get_logger

type RandomState = int | None

# Annotated types with Pydantic Field validation
type InitialStateLowType = Annotated[
    int,
    Field(
        default=0,
        help="Lower bound for initial state generation."
    )
]

type InitialStateHighType = Annotated[
    int,
    Field(
        default=1,
        help="Upper bound for initial state generation."
    )
]

type NSeriesType = Annotated[
    int,
    Field(
        default=100,
        gt=0,
        help="Number of spiral series to generate."
    )
]

type NTimestampsType = Annotated[
    int,
    Field(
        default=1000,
        gt=0,
        help="Number of time steps in each series."
    )
]

type TStepType = Annotated[
    float,
    Field(
        default=0.1,
        gt=0.0,
        help="Time step size for series generation."
    )
]

type DecayType = Annotated[
    float,
    Field(
        default=-0.1,
        help="Exponential decay rate (should be negative for stable spirals)."
    )
]

type FrequencyType = Annotated[
    float,
    Field(
        default=1.0,
        gt=0.0,
        help="Frequency for sinusoidal components."
    )
]

type NoiseType = Annotated[
    tuple[float, float] | None,
    Field(
        default=None,
        help="Noise parameters as (mean, std) tuple, or None for no noise."
    )
]

type SamplesType = Annotated[
    int,
    Field(
        gt=0,
        help="Number of samples for dataset split."
    )
]

type BatchSizeType = Annotated[
    int,
    Field(
        default=32,
        gt=0,
        help="Batch size for data loaders."
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
    return time, all_states


class StableSpirals(L.LightningDataModule):

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
        super(StableSpirals, self).__init__()

        self.initial_state_low = initial_state_low
        self.initial_state_high = initial_state_high
        self.n_timestamps = n_timestamps
        self.t_step = t_step
        self.x_decay = x_decay
        self.y_decay = y_decay
        self.x_freq = x_freq
        self.y_freq = y_freq
        self.batch_size = batch_size
        self.x_noise = x_noise
        self.y_noise = y_noise
        self.random_state = random_state

        self.train_series, self.train_datasetm, self._train_dataloader = self.generate_dataloader(train_samples, self.random_state)
        self.val_series, self.val_dataset, self._val_dataloader = self.generate_dataloader(val_samples, self.random_state+1)
        self.test_series, self.test_dataset, self._test_dataloader = self.generate_dataloader(test_samples, self.random_state+2)

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
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._test_dataloader
