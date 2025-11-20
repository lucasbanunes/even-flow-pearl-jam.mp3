import lightning as L
from sklearn.datasets import make_moons
import torch


class MoonsDataset(L.LightningDataModule):

    def __init__(self,
                 train_samples: int = 1000,
                 val_samples: int = 200,
                 test_samples: int = 200,
                 noise: float = 0.1,
                 batch_size: int = 32,
                 random_state: int = 42):
        super(MoonsDataset, self).__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.noise = noise
        self.batch_size = batch_size
        self.random_state = random_state
        self.train_moons, _ = make_moons(n_samples=self.train_samples,
                                         noise=self.noise, random_state=self.random_state)
        self.train_moons = torch.from_numpy(self.train_moons.astype('float32'))
        self.train_dataset = torch.utils.data.TensorDataset(
            self.train_moons
        )
        self._train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )
        self.val_moons, _ = make_moons(n_samples=self.val_samples,
                                       noise=self.noise, random_state=self.random_state + 1)
        self.val_moons = torch.from_numpy(self.val_moons.astype('float32'))
        self.val_dataset = torch.utils.data.TensorDataset(
            self.val_moons
        )
        self._val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )
        self.test_moons, _ = make_moons(n_samples=self.test_samples,
                                        noise=self.noise, random_state=self.random_state + 2)
        self.test_moons = torch.from_numpy(self.test_moons.astype('float32'))
        self.test_dataset = torch.utils.data.TensorDataset(
            self.test_moons
        )
        self._test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )

        def train_dataloader(self) -> torch.utils.data.DataLoader:
            return self._train_dataloader

        def val_dataloader(self) -> torch.utils.data.DataLoader:
            return self._val_dataloader

        def test_dataloader(self) -> torch.utils.data.DataLoader:
            return self._test_dataloader
