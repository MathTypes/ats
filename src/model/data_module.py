#import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import logging
from torch.utils.data import DataLoader
import logging
from datasets import (
    generate_stock_tokens,
    generate_stock_returns,
    tabular_to_sliding_dataset
)
import timeseries_dataset
import timeseries_utils
from pytorch_forecasting import Baseline, NHiTS, DeepAR, TimeSeriesDataSet

eval_batch_size = 10

class TransformerDataModule(pl.LightningDataModule):
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
        self.generate_data()

    def generate_data(self):
        if self.dataset == "stock_tokens":
            _tup = generate_stock_tokens()
        elif self.dataset == "stock_returns":
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
        if self.dataset == "stock_returns":
            logging.info(f"before:X_train:{self.X_train.shape}")
            self.X_train = self.X_train.permute(0, 2, 1)
            logging.info(f"after:X_train:{self.X_train.shape}")
            self.y_train = self.y_train.permute(0, 2, 1)
            self.X_val = self.X_val.permute(0, 2, 1)
            self.y_val = self.y_val.permute(0, 2, 1)
            self.X_test = self.X_test.permute(0, 2, 1)
            self.y_test = self.y_test.permute(0, 2, 1)
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

class LSTMDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=64, n_past=48, n_future=48):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_past = n_past
        self.n_future = n_future
        self.generate_data()

    def generate_data(self):
        if self.dataset == "stock_returns":
            _tup = generate_stock_returns(self.n_past, self.n_future)
        else:
            raise KeyError(f"Not supported dataset: {self.dataset}.")

        X_train, y_train, X_val, y_val, X_test, y_test = _tup

        self.X_train = torch.from_numpy(X_train)
        self.y_train = torch.from_numpy(y_train)
        self.X_val = torch.from_numpy(X_val)
        self.y_val = torch.from_numpy(y_val)
        self.X_test = torch.from_numpy(X_test)
        self.y_test = torch.from_numpy(y_test)

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


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config, train_data):
        super().__init__()
        self.train_data = train_data
        val_idx = int(len(train_data) * 0.7)
        tst_idx = max(int(len(train_data) * 0.8), len(train_data) - 2048)
        n_past = config['context_length']
        n_future = config['prediction_length']
        X_train, y_train, X_val, y_val, X_test, y_test = tabular_to_sliding_dataset(
            train_data[["hour_of_day", "volume_back", "close_back"]].values,
            validation_idx=val_idx,
            test_idx=tst_idx,
            n_past=n_past,
            n_future=n_future
        )
        self.X_train = torch.from_numpy(X_train)
        self.X_val = torch.from_numpy(X_val)
        self.X_test = torch.from_numpy(X_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_val = torch.from_numpy(y_val)
        self.y_test = torch.from_numpy(y_test)
        logging.info(f"train_data:{train_data}")
        self.training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="close_back",
            group_ids=["ticker"],
            min_encoder_length=config['context_length'],  # keep encoder length long (as it is in the validation set)
            max_encoder_length=config['context_length'],
            min_prediction_length=config['prediction_length'],
            max_prediction_length=config['prediction_length'],
            allow_missing_timesteps=True,
            time_varying_known_reals=["time_idx", "hour_of_day"],
            time_varying_unknown_reals=["close_back", "volume_back"],
            categorical_encoders={},
        )
        logging.info(f"train_data:{train_data.describe()}")
        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self.validation = TimeSeriesDataSet.from_dataset(self.training, train_data[val_idx:tst_idx])
        self.test = TimeSeriesDataSet.from_dataset(self.training, train_data[tst_idx:])
        # create dataloaders for model
        self.batch_size = 128*20  # set this between 32 to 128
        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass
        
    def train_dataloader(self):
        train_dataloader = self.training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=10, pin_memory=True, drop_last=False)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0, pin_memory=True, drop_last=False)
        return val_dataloader

    
    def test_dataloader(self):
        test_dataloader = self.test.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0, pin_memory=True, drop_last=False)
        return test_dataloader
