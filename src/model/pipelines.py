import torch
import logging
import numpy as np
from utils import Pipeline
from timeseries_transformer import TimeSeriesTFT
from data_module import AtsDataModule
from models import (
    AttentionEmbeddingLSTM
)

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
    def __init__(self, dataset="sine_wave"):
        super().__init__()
        self.dataset = dataset

    def create_model(self):
        self.data_module = AtsDataModule("stock_returns",
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
    def __init__(self, dataset="sine_wave"):
        super().__init__()
        self.dataset = dataset

    def create_model(self, device):
        self.data_module = AtsDataModule("stock_returns")
        X_train = self.data_module.X_train
        y_train = self.data_module.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]
        logging.info(f"features:{features}")
        logging.info(f"mini_batch:{mini_batch}")
        model = AttentionEmbeddingLSTM(
            linear_channel=features,
            period_channel=(mini_batch - features),
            input_channel=mini_batch,
            input_size=X_train.shape[2],
            out_size=y_train.shape[-1]
        )
        self.model = model.to(device)