from typing import Annotated
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pydantic import Field
import lightning as L
import mlflow


from ..pydantic import MLFlowLoggedModel


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


type DataDirType = Annotated[
    Path,
    Field(
        description="Directory where MNIST data is/will be stored",
    )
]

type BatchSizeType = Annotated[
    int,
    Field(
        description="Batch size for data loading",
    )
]

type RandomSeedType = Annotated[
    int,
    Field(
        description="Random seed for data splitting and shuffling",
    )
]


class MNISTDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: DataDirType,
                 batch_size: BatchSizeType,
                 random_seed: RandomSeedType = 42,
                 flatten: bool = False,
                 train_samples: int = -1,
                 val_samples: int = -1,
                 test_samples: int = -1):
        super().__init__()
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.flatten = flatten
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
        if self.flatten:
            transformations.append(transforms.Lambda(torch.flatten))
        self.transform = transforms.Compose(transformations)

        mnist_full = MNIST(self.data_dir, train=True,
                           transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(self.random_seed)
        )
        if self.train_samples > 0:
            self.mnist_train = torch.utils.data.Subset(
                self.mnist_train, range(self.train_samples))
        if self.val_samples > 0:
            self.mnist_val = torch.utils.data.Subset(
                self.mnist_val, range(self.val_samples))

        self.mnist_test = MNIST(
            self.data_dir, train=False, transform=self.transform)

        if self.test_samples > 0:
            self.mnist_test = torch.utils.data.Subset(
                self.mnist_test, range(self.test_samples))

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class MNISTDataset(MLFlowLoggedModel):

    data_dir: DataDirType = Path("./.cache/mnist")
    batch_size: BatchSizeType = 32
    random_seed: RandomSeedType = 42
    flatten: bool = False
    train_samples: int = -1
    val_samples: int = -1
    test_samples: int = -1

    def model_post_init(self, context):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return super().model_post_init(context)

    def as_lightning_datamodule(self) -> MNISTDataModule:
        data_dir = self.data_dir / 'lightning_module'
        data_dir.mkdir(parents=True, exist_ok=True)
        return MNISTDataModule(
            data_dir=data_dir,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            flatten=self.flatten,
            train_samples=self.train_samples,
            val_samples=self.val_samples,
            test_samples=self.test_samples,
        )

    def _to_mlflow(self, prefix=''):
        if prefix:
            prefix += '.'
        mlflow.log_param(f'{prefix}data_dir', str(self.data_dir))
        mlflow.log_param(f'{prefix}batch_size', self.batch_size)
        mlflow.log_param(f'{prefix}random_seed', self.random_seed)
        mlflow.log_param(f'{prefix}flatten', self.flatten)
        mlflow.log_param(f'{prefix}train_samples', self.train_samples)
        mlflow.log_param(f'{prefix}val_samples', self.val_samples)
        mlflow.log_param(f'{prefix}test_samples', self.test_samples)

    @classmethod
    def _from_mlflow(cls, mlflow_run, prefix='', **kwargs):
        if prefix:
            prefix += '.'
        kwargs['data_dir'] = mlflow_run.data.params.get(f'{prefix}data_dir',
                                                        cls.model_fields['data_dir'].default)
        kwargs['batch_size'] = int(mlflow_run.data.params.get(f'{prefix}batch_size',
                                                              cls.model_fields['batch_size'].default))
        kwargs['random_seed'] = int(mlflow_run.data.params.get(f'{prefix}random_seed',
                                                               cls.model_fields['random_seed'].default))
        kwargs['flatten'] = mlflow_run.data.params.get(f'{prefix}flatten',
                                                       cls.model_fields['flatten'].default)
        kwargs['flatten'] = kwargs['flatten'] in ['True', 'true', True]
        kwargs['train_samples'] = int(mlflow_run.data.params.get(f'{prefix}train_samples',
                                                                 cls.model_fields['train_samples'].default))
        kwargs['val_samples'] = int(mlflow_run.data.params.get(f'{prefix}val_samples',
                                                               cls.model_fields['val_samples'].default))
        kwargs['test_samples'] = int(mlflow_run.data.params.get(f'{prefix}test_samples',
                                                                cls.model_fields['test_samples'].default))
        return kwargs
