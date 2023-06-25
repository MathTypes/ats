import torch
import logging
import numpy as np
from utils import Pipeline
from timeseries_transformer import TimeSeriesTFT
from data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from models import (
    AttentionEmbeddingLSTM
)
import trainer_nhits_with_dp as nhits

torch.manual_seed(0)
np.random.seed(0)

target_col_name = ["OpenPct", "HighPct", "LowPct", "ClosePct", "VolumePct"]

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
enc_seq_len = 153 # length of input given to encoder
batch_first = False
forecast_window = 24
# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = target_col_name + exogenous_vars
input_size = len(input_variables)

torch.set_float32_matmul_precision('medium')

class TFTPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = TransformerDataModule("stock_returns",
                                                 output_sequence_length=forecast_window,
                                                 batch_size=2048)
        X_train = self.data_module.X_train
        dev = "cuda"

        logging.info(f"X_train:{X_train.shape}, dev:{dev}")
        self.model = TimeSeriesTFT(
            input_size=5,
            dim_val=16,
            dec_seq_len=enc_seq_len,
            batch_first=batch_first,
            forecast_window=forecast_window,
            num_predicted_features=1,
            device=dev
        ).float().to(dev)


class AttentionEmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = LSTMDataModule("stock_returns", batch_size=32, n_past=48, n_future=12)
        X_train = self.data_module.X_train
        y_train = self.data_module.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]
        logging.info(f"features:{features}")
        logging.info(f"mini_batch:{mini_batch}")
        logging.info(f"y_train:{y_train.shape}")
        linear_channel = 1
        model = AttentionEmbeddingLSTM(
            input_features=features,
            linear_channel=linear_channel,
            period_channel=(mini_batch - linear_channel),
            input_size=mini_batch,
            out_size=y_train.shape[-1],
            out_values = 1,
            hidden_size=4
        )
        self.model = model.to(self.device, non_blocking=True)


class TimeSeriesPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = nhits.get_data_module(self.config)
        self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        nhits.run_tune(config, study_name)

class TemporalFusionTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = nhits.get_data_module(self.config)
        self.model = nhits.get_tft_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class PatchTstTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = nhits.get_data_module(self.config)
        self.model = nhits.get_patch_tst_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass
