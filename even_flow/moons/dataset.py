from typing import Any, Self, Annotated
import lightning as L
from sklearn.datasets import make_moons
import torch
from pydantic import Field
import mlflow
from mlflow.entities import Run


from ..mlflow import MLFlowLoggedClass

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


class MoonsDataset(L.LightningDataModule, MLFlowLoggedClass):
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
        super(MoonsDataset, self).__init__()

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.noise = noise
        self.batch_size = batch_size
        self.random_state = random_state

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the training DataLoader.
        Returns:
            Training DataLoader with moon-shaped data
        """
        if self._train_dataloader is None:
            self._train_dataloader = self.generate_dataloader(
                n_samples=self.train_samples,
                random_state=self.random_state,
                float_label=True)
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the validation DataLoader.

        Returns:
            Validation DataLoader with moon-shaped data
        """
        if self._val_dataloader is None:
            self._val_dataloader = self.generate_dataloader(
                n_samples=self.val_samples,
                random_state=self.random_state + 1)
        return self._val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create and return the test DataLoader.

        Returns:
            Test DataLoader with moon-shaped data
        """
        if self._test_dataloader is None:
            self._test_dataloader = self.generate_dataloader(
                n_samples=self.test_samples,
                random_state=self.random_state + 2)
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
        labels = labels.reshape(-1, 1)
        if float_label:
            labels = labels.astype('float32')
        else:
            labels = labels.astype('int64')
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(moons.astype('float32')),
                                                 torch.from_numpy(labels))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size)
        return dataloader

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

    def to_mlflow(self, prefix: str = "") -> dict[str, Any]:
        """Log dataset parameters to MLflow."""
        if prefix:
            prefix += "."
        mlflow.log_params({
            f"{prefix}train_samples": self.train_samples,
            f"{prefix}val_samples": self.val_samples,
            f"{prefix}test_samples": self.test_samples,
            f"{prefix}noise": self.noise,
            f"{prefix}batch_size": self.batch_size,
            f"{prefix}random_state": self.random_state,
        })

    @classmethod
    def from_mlflow(cls, mlflow_run: Run, prefix: str = "") -> Self:
        """Create an instance from MLflow logged parameters."""
        if prefix:
            prefix += "."
        return cls(
            train_samples=int(mlflow_run.data.params[f'{prefix}train_samples']),
            val_samples=int(mlflow_run.data.params[f'{prefix}val_samples']),
            test_samples=int(mlflow_run.data.params[f'{prefix}test_samples']),
            noise=float(mlflow_run.data.params[f'{prefix}noise']),
            batch_size=int(mlflow_run.data.params[f'{prefix}batch_size']),
            random_state=int(mlflow_run.data.params[f'{prefix}random_state']),
        )
