from typing import Any, Self, Annotated
import lightning as L
from sklearn.datasets import make_moons
import torch
from pydantic import Field
import mlflow
from mlflow.entities import Run

from ..pydantic import MLFlowLoggedModel

type RandomState = int | None

# Annotated types with Pydantic Field validation
type TrainSamplesType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of training samples to generate."
    )
]

type ValSamplesType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of validation samples to generate."
    )
]

type TestSamplesType = Annotated[
    int,
    Field(
        gt=0,
        description="Number of test samples to generate."
    )
]

type NoiseType = Annotated[
    float,
    Field(
        ge=0.0,
        description="Amount of homoscedastic noise to add to the data."
    )
]

type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for data loaders."
    )
]


class MoonsDataModule(L.LightningDataModule):
    """
    Lightning DataModule for generating 2D moon-shaped datasets.

    This dataset creates two interleaving half circles (moons) that are commonly
    used for testing non-linear classification algorithms. The dataset is generated
    using sklearn's make_moons function.

    Args:
        train_samples: Number of training samples to generate
        val_samples: Number of validation samples to generate
        test_samples: Number of test samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        batch_size: Batch size for all data loaders
        random_state: Random state for reproducible data generation
    """

    def __init__(self,
                 train_samples: TrainSamplesType = 1000,
                 val_samples: ValSamplesType = 200,
                 test_samples: TestSamplesType = 200,
                 noise: NoiseType = 0.1,
                 batch_size: BatchSizeType = 32,
                 random_state: RandomState = 42):
        super().__init__()

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.noise = noise
        self.batch_size = batch_size
        self.random_state = random_state

        self.train_X, self.train_y, self._train_dataloader = self.generate_dataloader(
            n_samples=self.train_samples,
            random_state=self.random_state,
            float_label=True)

        self.val_X, self.val_y, self._val_dataloader = self.generate_dataloader(
            n_samples=self.val_samples,
            random_state=self.random_state + 1)

        self.test_X, self.test_y, self._test_dataloader = self.generate_dataloader(
            n_samples=self.test_samples,
            random_state=self.random_state + 2)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the training DataLoader.
        Returns:
            Training DataLoader with moon-shaped data
        """
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the validation DataLoader.

        Returns:
            Validation DataLoader with moon-shaped data
        """
        return self._val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the test DataLoader.

        Returns:
            Test DataLoader with moon-shaped data
        """
        return self._test_dataloader

    def generate_dataloader(self,
                            n_samples: int,
                            random_state: RandomState = 42,
                            float_label: bool = False) -> torch.utils.data.DataLoader:
        """
        Generate a DataLoader with moon-shaped data.

        Args:
            n_samples: Number of samples to generate
            random_state: Random state for reproducible generation

        Returns:
            DataLoader containing the generated moon dataset
        """
        moons, labels = make_moons(n_samples=n_samples,
                                   noise=self.noise,
                                   random_state=random_state)
        moons = torch.from_numpy(moons.astype('float32'))
        labels = labels.reshape(-1, 1)
        if float_label:
            labels = labels.astype('float32')
        else:
            labels = labels.astype('int64')
        labels = torch.from_numpy(labels)
        dataset = torch.utils.data.TensorDataset(moons,
                                                 labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size)
        return moons, labels, dataloader

    @classmethod
    def pydantic_before_validator(cls, v: Any) -> Self:
        """Pydantic validator to handle various input types."""
        if isinstance(v, cls):
            return v
        elif v is None:
            return cls()
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Cannot convert {type(v)} to {cls}.")

    @staticmethod
    def pydantic_plain_serializer(v: 'MoonsDataset') -> dict[str, Any]:
        """Pydantic serializer for converting to dictionary."""
        return {
            "train_samples": v.train_samples,
            "val_samples": v.val_samples,
            "test_samples": v.test_samples,
            "noise": v.noise,
            "batch_size": v.batch_size,
            "random_state": v.random_state,
        }


class MoonsDataset(MLFlowLoggedModel):
    train_samples: TrainSamplesType = 1000
    val_samples: ValSamplesType = 200
    test_samples: TestSamplesType = 200
    noise: NoiseType = 0.1
    batch_size: BatchSizeType = 32
    random_state: RandomState = 42

    def as_lightning_datamodule(self) -> MoonsDataModule:
        return MoonsDataModule(
            train_samples=self.train_samples,
            val_samples=self.val_samples,
            test_samples=self.test_samples,
            noise=self.noise,
            batch_size=self.batch_size,
            random_state=self.random_state
        )

    def _to_mlflow(self, prefix: str = ''):
        mlflow.log_param(f'{prefix}.train_samples', self.train_samples)
        mlflow.log_param(f'{prefix}.val_samples', self.val_samples)
        mlflow.log_param(f'{prefix}.test_samples', self.test_samples)
        mlflow.log_param(f'{prefix}.noise', self.noise)
        mlflow.log_param(f'{prefix}.batch_size', self.batch_size)
        mlflow.log_param(f'{prefix}.random_state', self.random_state)

    @classmethod
    def _from_mlflow(cls, mlflow_run: Run, prefix: str = '', **kwargs) -> dict[str, Any]:
        if prefix:
            prefix += '.'
        kwargs['train_samples'] = int(mlflow_run.data.params.get(
            f'{prefix}train_samples', cls.model_fields['train_samples'].default))
        kwargs['val_samples'] = int(mlflow_run.data.params.get(
            f'{prefix}val_samples', cls.model_fields['val_samples'].default))
        kwargs['test_samples'] = int(mlflow_run.data.params.get(
            f'{prefix}test_samples', cls.model_fields['test_samples'].default))
        kwargs['noise'] = float(mlflow_run.data.params.get(
            f'{prefix}noise', cls.model_fields['noise'].default))
        kwargs['batch_size'] = int(mlflow_run.data.params.get(
            f'{prefix}batch_size', cls.model_fields['batch_size'].default))
        kwargs['random_state'] = int(mlflow_run.data.params.get(
            f'{prefix}random_state', cls.model_fields['random_state'].default))
        return kwargs
