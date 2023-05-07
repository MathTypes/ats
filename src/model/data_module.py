import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import (
    generate_stock_returns,
)

class AtsDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.generate_data()

    def generate_data(self):
        if self.dataset == "stock_returns":
            _tup = generate_stock_returns()
        else:
            raise KeyError(f"Not supported dataset: {self.dataset}.")

        X_train, y_train, X_val, y_val, X_test, y_test = _tup

        self.X_train = torch.from_numpy(X_train)
        self.y_train = torch.from_numpy(y_train)
        self.X_val = torch.from_numpy(X_val)
        self.y_val = torch.from_numpy(y_val)
        self.X_test = torch.from_numpy(X_test)
        self.y_test = torch.from_numpy(y_test)
        self.batch_size = 256

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        #self.generate_data()
        pass
        
    def train_dataloader(self):
        train_wrapper = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_wrapper, batch_size=self.batch_size, pin_memory=True, num_workers=8)
    
    def val_dataloader(self):
        val_wrapper = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_wrapper, batch_size=self.batch_size, pin_memory=True, num_workers=8)
    
    def test_dataloader(self):
        test_wrapper = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_wrapper, batch_size=self.batch_size, pin_memory=True, num_workers=8)
