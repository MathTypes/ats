import pytorch_lightning as pl
import torch
import logging
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, AutoAugment, AutoAugmentPolicy
import logging
from datasets import (
    generate_stock_tokens,
)
import timeseries_dataset
import timeseries_utils


eval_batch_size = 10

class AtsDataModule(pl.LightningDataModule):
    def __init__(self, dataset,
                 dec_seq_len = 92,
                 enc_seq_len = 153,
                 output_sequence_length = 24,
                 step_size: int = 1,
                 batch_size = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.output_sequence_length = output_sequence_length
        self.window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
        self.step_size = step_size
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.val_transform = transforms.Compose(  
            [  
                transforms.ToTensor(),  
            ]  
        )  
        self.generate_data()

    def generate_data(self):
        if self.dataset == "stock_returns":
            _tup = generate_stock_tokens()
        else:
            raise KeyError(f"Not supported dataset: {self.dataset}.")

        X_train, X_val, X_test = _tup
        self.X_train = torch.from_numpy(X_train)
        self.X_val = torch.from_numpy(X_val)
        self.X_test = torch.from_numpy(X_test)
        logging.info(f"X_Train:{self.X_train.shape}")
        logging.info(f"X_eval:{self.X_val[0:2]}")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        #self.generate_data()
        pass
        
    def train_dataloader(self):
        self.training_indices = timeseries_utils.get_indices_entire_sequence(
            data=self.X_train.numpy(), 
            window_size=self.window_size, 
            step_size=self.step_size)
        train_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_train,
            indices=self.training_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.output_sequence_length
            )
        return DataLoader(train_wrapper, batch_size=self.batch_size,
                          pin_memory=True, num_workers=12, shuffle=True)
    
    def val_dataloader(self):
        self.val_indices = timeseries_utils.get_indices_entire_sequence(
            data=self.X_val.numpy(), 
            window_size=self.window_size, 
            step_size=self.step_size)
        val_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_val,
            indices=self.val_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.output_sequence_length
            )
        return DataLoader(val_wrapper, batch_size=self.batch_size, pin_memory=True,
                          num_workers=8)
    
    def test_dataloader(self):
        self.test_indices = timeseries_utils.get_indices_entire_sequence(
            data=self.X_test.numpy(), 
            window_size=self.window_size, 
            step_size=self.step_size)
        test_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_test,
            indices=self.val_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.output_sequence_length
            )
        return DataLoader(test_wrapper, batch_size=self.batch_size, pin_memory=True,
                          num_workers=8)
