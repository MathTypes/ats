import datetime
from io import BytesIO
import logging

import pandas_market_calendars as mcal
import numpy as np
# find optimal learning rate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, PatchTstTransformer, PatchTstTftTransformer, PatchTftSupervised
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, MAPCSE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.utils import create_mask, detach, to_list
import pytz
import plotly.graph_objects as go
import PIL
from plotly.subplots import make_subplots
from timeseries_transformer import TimeSeriesTFT
import torch
import wandb

from data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from models import (
    AttentionEmbeddingLSTM
)
import model_utils
from utils import Pipeline
import viz_utils

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
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        self.train_start_date = datetime.datetime.strptime(config.job.train_start_date,"%Y-%m-%d")
        self.test_end_date = datetime.datetime.strptime(config.job.test_end_date,"%Y-%m-%d")
        self.data_module = model_utils.get_data_module(self.config.dateset.base_dir,
                                                       self.train_start_date,
                                                       self.test_end_date,
                                                       self.targets)
        loss_per_head = model_utils.create_loss_per_head(self.heads, self.device, self.config.model.prediction_length)
        self.model = model_utils.get_nhits_model(self.config, self.data_module, loss_per_head["returns_prediction"]["loss"])
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        nhits.run_tune(self.config, study_name)

    def test_model(self):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class TemporalFusionTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_tft_model(self.config, self.data_module)
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
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

class PatchTstTftPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_tft_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass
    

class PatchTftSupervisedPipeline(Pipeline):
    def __init__(self, dataset="fut", config=None):
        super().__init__(config)
        self.dataset = dataset
        self.config = config
        self.init_env()

    def init_env(self):
        self.train_start_date = datetime.datetime.strptime(self.config.job.train_start_date,"%Y-%m-%d")
        self.test_start_date = datetime.datetime.strptime(self.config.job.test_start_date,"%Y-%m-%d")
        self.test_end_date = datetime.datetime.strptime(self.config.job.test_end_date,"%Y-%m-%d")
        self.max_lags = self.config.job.max_lags
        if self.config.job.mode == "train":
            self.train_start_date = datetime.datetime.strptime(self.config.job.train_start_date,"%Y-%m-%d")
        elif self.config.job.mode == "test":
            self.data_start_date = self.test_start_date - datetime.timedelta(days=self.max_lags)
            self.market_cal = mcal.get_calendar(self.config.job.market)
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        logging.info(f"head:{self.heads}, targets:{self.targets}")
        start_date = self.train_start_date
        if not start_date:
            start_date = self.data_start_date
        logging.info(f"start_date:{start_date}, test_start_date:{self.test_start_date}, test_end_date:{self.test_end_date}")
        self.data_module = model_utils.get_data_module(self.config,
                                                       self.config.dataset.base_dir,
                                                       start_date,
                                                       self.test_start_date,
                                                       self.test_end_date,
                                                       self.targets,
                                                       self.config.dataset.model_tickers,
                                                       self.config.dataset.time_interval)
        
    def create_model(self, checkpoint):
        self.model = model_utils.get_patch_tft_supervised_model(self.config, self.data_module, self.heads)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        if checkpoint:
            self.model = self.model.load_from_checkpoint(checkpoint)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        #self.data_module = nhits.get_data_module(self.config)
        #self.model = nhits.get_model(self.config, self.data_module)
        #self.trainer = nhits.get_trainer(self.config, self.data_module)
        #self.model = self.model.to(self.device, non_blocking=True)
        #nhits.run_tune(config, study_name)
        pass

    def test_model(self):
        test_dates = self.market_cal.valid_days(start_date=self.test_start_date, end_date=self.test_end_date)
        train_dataset = self.data_module.training
        train_data = self.data_module.train_data        
        future_data = self.data_module.eval_data
        
        logging.info(f"future_data:{future_data.iloc[:1]}")
        wandb_logger = WandbLogger(project='ATS', log_model=True)
        last_time_idx = train_data.iloc[-1]["time_idx"]
        for test_date in test_dates:
            schedule = self.market_cal.schedule(start_date=test_date, end_date=test_date)
            logging.info(f"sod {test_date}")
            time_range = mcal.date_range(schedule, frequency='30M')
            max_prediction_length = self.config.model.prediction_length
            trainer_kwargs = {"logger" : wandb_logger}
            for utc_time in time_range:
                nyc_time = utc_time.astimezone(pytz.timezone('America/New_York'))
                # 1. prepare prediction with latest prices
                # 2. run inference to get returns and new positions
                # 3. update PNL and positions
                logging.info(f"running step {nyc_time}")
                new_data = future_data[(future_data.timestamp==nyc_time.timestamp()) & (future_data.ticker=="ES")]
                if new_data.empty:
                    continue
                last_time_idx += 1
                new_data["time_idx"] = last_time_idx
                train_dataset.add_new_data(new_data)
                new_prediction_data = train_dataset.filter(lambda x: (x.time_idx_last == last_time_idx))
                #logging.info(f"new_prediction_data:{new_prediction_data}")
                new_raw_predictions = self.model.predict(new_prediction_data, mode="raw", return_x=True,
                                                         trainer_kwargs=trainer_kwargs)
                #logging.info(f"new_raw_predictions:{new_raw_predictions}")
            logging.info(f"eod {test_date}")


    def eval_model(self):
        #self.data_module = nhits.get_data_module(self.config)
        # calcualte metric by which to display
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project='ATS', log_model=True)        
        trainer_kwargs = {'logger':wandb_logger}
        logging.info(f"rows:{len(self.data_module.eval_data)}")
        data_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type="evaluation")
        metrics = SMAPE(reduction="none").to(self.device)
        data_table = viz_utils.create_example_viz_table(self.model.to(self.device),
                                                        self.data_module.val_dataloader(),
                                                        self.data_module.eval_data, metrics,
                                                        self.config.job.eval_top_k)
        data_artifact.add(data_table, "eval_data")

        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()
        
