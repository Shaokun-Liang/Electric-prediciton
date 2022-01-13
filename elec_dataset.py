import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Optional, Literal
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import get_array_data
from config import config


class ElectricityDataModule(pl.LightningDataModule):
    def __init__(self):
        super(ElectricityDataModule, self).__init__()
        self.batch_size = config.batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        train_arr, val_arr, test_arr = get_array_data(decom_method=config.decom_method, dim=3)
        if stage in (None, 'fit'):
            X_train, y_train = torch.from_numpy(train_arr[:, :, :-1]), torch.from_numpy(train_arr[:, :, -1])
            X_val, y_val = torch.from_numpy(val_arr[:, :, :-1]), torch.from_numpy(val_arr[:, :, -1])
            self.train = TensorDataset(X_train, y_train)
            self.val = TensorDataset(X_val, y_val)
        if stage in (None, 'test'):
            X_test, y_test = torch.from_numpy(test_arr[:, :, :-1]), torch.from_numpy(test_arr[:, :, -1])
            self.test = TensorDataset(X_test, y_test)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size)
