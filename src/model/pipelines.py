import torch
import logging
import numpy as np
from utils import Pipeline
from timeseries_transformer import TimeSeriesTFT
from data_module import AtsDataModule

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
        self.data_module = AtsDataModule("stock_returns")
        X_train = self.data_module.X_train
        logging.info(f"X_train:{X_train.shape}")
        self.model = TimeSeriesTFT(
            input_size=5,
            dec_seq_len=enc_seq_len,
            batch_first=batch_first,
            num_predicted_features=5
        ).float().to("cuda:0")
