from collections import defaultdict
import datetime
from dateutil import parser
import gc
import logging
from typing import List

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
    # cum_returns,
)
# find optimal learning rate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
import optuna

import pytz
import plotly.graph_objects as go
import PIL
from plotly.subplots import make_subplots
from pytorch_forecasting import (
    Baseline,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    PatchTstTransformer,
    PatchTstTftTransformer,
    PatchTftSupervised,
)
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    MAPCSE,
    RMSE,
    SMAPE,
    PoissonLoss,
    QuantileLoss,
)
from pytorch_forecasting.utils import create_mask, detach, to_list
from pytorch_forecasting.models.patch_tft_supervised.tuning import (
    optimize_hyperparameters,
)
import torch
import wandb

from ats.app.env_mgr import EnvMgr
from ats.calendar import market_time
from ats.market_data import market_data_mgr
from ats.market_data.data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from ats.model.models import AttentionEmbeddingLSTM
from ats.model.timeseries_transformer import TimeSeriesTFT
from ats.model import model_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.prediction import prediction_utils
from ats.trading.trader import Trader
from ats.util.profile import profile
from ats.util import trace_utils

torch.manual_seed(0)
np.random.seed(0)

target_col_name = ["OpenPct", "HighPct", "LowPct", "ClosePct", "VolumePct"]

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
enc_seq_len = 153  # length of input given to encoder
batch_first = False
forecast_window = 24
# Define input variables
exogenous_vars = (
    []
)  # should contain strings. Each string must correspond to a column name
input_variables = target_col_name + exogenous_vars
input_size = len(input_variables)

torch.set_float32_matmul_precision("medium")


class TFTPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = TransformerDataModule(
            "stock_returns", output_sequence_length=forecast_window, batch_size=2048
        )
        X_train = self.data_module.X_train
        dev = "cuda"

        logging.info(f"X_train:{X_train.shape}, dev:{dev}")
        self.model = (
            TimeSeriesTFT(
                input_size=5,
                dim_val=16,
                dec_seq_len=enc_seq_len,
                batch_first=batch_first,
                forecast_window=forecast_window,
                num_predicted_features=1,
                device=dev,
            )
            .float()
            .to(dev)
        )


class AttentionEmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave", device=None):
        super().__init__(device)
        self.dataset = dataset

    def create_model(self):
        self.data_module = LSTMDataModule(
            "stock_returns", batch_size=32, n_past=48, n_future=12
        )
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
            out_values=1,
            hidden_size=4,
        )
        self.model = model.to(self.device, non_blocking=True)


class TimeSeriesPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        self.train_start_date = datetime.datetime.strptime(
            config.job.train_start_date, "%Y-%m-%d"
        )
        self.test_end_date = datetime.datetime.strptime(
            config.job.test_end_date, "%Y-%m-%d"
        )
        self.data_module = model_utils.get_data_module(
            self.config.dateset.base_dir,
            self.train_start_date,
            self.test_end_date,
            self.targets,
        )
        loss_per_head = model_utils.create_loss_per_head(
            self.heads, self.device, self.config.model.prediction_length
        )
        self.model = model_utils.get_nhits_model(
            self.config, self.data_module, loss_per_head["returns_prediction"]["loss"]
        )
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        # self.data_module = nhits.get_data_module(self.config)
        # self.model = nhits.get_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        # self.model = self.model.to(self.device, non_blocking=True)
        nhits.run_tune(self.config, study_name)

    def test_model(self):
        # self.data_module = nhits.get_data_module(self.config)
        # self.model = nhits.get_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        # self.model = self.model.to(self.device, non_blocking=True)
        # nhits.run_tune(config, study_name)
        pass


class TemporalFusionTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_tft_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        # self.data_module = nhits.get_data_module(self.config)
        # self.model = nhits.get_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        # self.model = self.model.to(self.device, non_blocking=True)
        # nhits.run_tune(config, study_name)
        pass


class PatchTstTransformerPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name):
        # self.data_module = nhits.get_data_module(self.config)
        # self.model = nhits.get_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        # self.model = self.model.to(self.device, non_blocking=True)
        # nhits.run_tune(config, study_name)
        pass


class PatchTstTftPipeline(Pipeline):
    def __init__(self, dataset="fut", device=None, config=None):
        super().__init__(device)
        self.dataset = dataset
        self.config = config

    def create_model(self):
        self.data_module = model_utils.get_data_module(self.config)
        self.model = model_utils.get_patch_tst_tft_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, study_name, config):
        # self.data_module = nhits.get_data_module(self.config)
        # self.model = nhits.get_model(self.config, self.data_module)
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        # self.model = self.model.to(self.device, non_blocking=True)
        # nhits.run_tune(config, study_name)
        pass

class TftParams(object):
    """Esta clase contiene los parametros necesarios del modelo"""
    def __init__(self):
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 120
        self.LEARNING_RATE = 0.0016879719644926218
        self.PATIENCE = 30
        self.DROPOUT = 0.12717796990901462
        self.HIDDEN_LAYER_SIZE = 99
        self.EMBEDDING_DIMENSION = 79
        self.NUM_LSTM_LAYERS = 2
        self.NUM_ATTENTION_HEADS = 1
        self.QUANTILES = [0.1, 0.5, 0.9]
        self.GRADIENT_CLIP_VAL = 0.01002874085767653
    
        self.ENCODER_LEN_MIN = 252 #1 anio
        self.ENCODER_LEN_MAX = 252 #1 anio
        self.DECODER_LEN = 63 # 1 trimestre
        self.TIME_IDX = 'Time_idx'
        self.TARGET = ['Close', 'Volume']
        self.GROUP_ID = ["Ticker"]
        self.STATIC_CATEGORICALS = ["Ticker"]
        self.TIME_KNOW_CATEGORICALS = ['Date_day', 'Date_month', 'Date_day_week', 'Date_day_year']
        self.TIME_KNOW_REALS = ['Time_idx']
        self.TIME_UNKNOW_CATEGORICALS = []
        self.TIME_UNKNOW_REALS = ["Close", "Open", "Volume"]

class PatchTftSupervisedPipeline(Pipeline):
    def __init__(self, dataset="fut", config=None):
        super().__init__(config)
        self.dataset = dataset
        self.config = config
        self.env_mgr = EnvMgr(config)
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        logging.info(f"head:{self.heads}, targets:{self.targets}")

        self.data_module = model_utils.get_data_module(
            self.config,
            self.config.dataset.base_dir,
            self.env_mgr,
            self.targets,
            self.config.dataset.model_tickers,
            self.config.dataset.time_interval,
            simulation_mode=True,
        )

    def create_model(self, checkpoint):
        self.model = model_utils.get_patch_tft_supervised_model(
            self.config, self.data_module, self.heads
        )
        # self.trainer = nhits.get_trainer(self.config, self.data_module)
        if checkpoint:
            self.model = self.model.load_from_checkpoint(checkpoint)
        self.model = self.model.to(self.device, non_blocking=True)

    def tune_model(self, run_id):
        study_name = f"tft_study_{run_id}"
        opt = optuna.create_study(direction="minimize",
                                  pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                  study_name=study_name)
        kwargs = {"loss": QuantileLoss(quantiles=TftParams().QUANTILES)}
        study = optimize_hyperparameters(
            self.data_module.train_dataloader(),
            self.data_module.val_dataloader(),
            self.heads,
            self.targets,
            self.device,
            model_path="optuna_test",
            n_trials=20,
            max_epochs=1,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30,
                                accelerator='gpu', devices=-1,
                                callbacks=[]),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            study=opt, 
            timeout=60*60*2, #2 horas
            **kwargs
        )
        print('best_trial', study.best_trial.params)

    def test_model(self):
        test_dates = self.market_cal.valid_days(
            start_date=self.env_mgr.test_start_date, end_date=self.env_mgr.test_end_date
        )
        train_dataset = self.data_module.training
        train_data = self.data_module.train_data
        future_data = self.data_module.test_data

        #logging.info(f"train_data:{train_data.iloc[-2:]}")
        #logging.info(f"future_data:{future_data.iloc[:2]}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        data_artifact = wandb.Artifact(f"run_{wandb.run.id}_pnl_viz", type="pnl_viz")
        column_names = [
            "ticker",
            "time",
            "time_idx",
            "day_of_week",
            "hour_of_day",
            "year",
            "month",
            "day_of_month",
            "price_img",
            "act_close_pct_max",
            "act_close_pct_min",
            "close_back_cumsum",
            "time_str",
            "pred_time_idx",
            "pred_close_pct_max",
            "pred_close_pct_min",
            "img",
            "error_max",
            "error_min",
            "rmse",
            "mae",
            "initial_pos",
            "new_pos",
            "delta_pos",
            "current_px",
            "pnl",
        ]
        data_table = wandb.Table(columns=column_names, allow_mixed_types=True)
        target_size = len(self.targets) if isinstance(self.targets, List) else 1

        md_mgr = market_data_mgr.MarketDataMgr(self.config, self.market_cal)
        trader = Trader(md_mgr, self.model, wandb_logger, target_size, future_data,
                        train_data, train_dataset, self.config, self.market_cal)
        for test_date in test_dates:
            schedule = self.market_cal.schedule(
                start_date=test_date, end_date=test_date
            )
            time_range = mcal.date_range(schedule, frequency="30M")
            logging.info(f"sod {test_date}, schedule:{time_range}")
            for utc_time in time_range:
                row = trader.on_interval(utc_time)
                logging.info(f"got row:{row}")
                if row:
                    logging.info(f"row:{row}")
                    data_table.add_data(
                        row["ticker"],  # 0 ticker
                        row["dm"],  # 1 time
                        row["time_idx"],  # 2 time_idx
                        row["day_of_week"],  # 3 day of week
                        row["hour_of_day"],  # 4 hour of day
                        row["year"],  # 5 year
                        row["month"],  # 6 month
                        row["day_of_month"],  # 7 day_of_month
                        row["image"],  # 8 image
                        row["y_close_cum_max"],  # 9 max
                        row["y_close_cum_min"],  # 10 min
                        row["close_back_cumsum"],  # 11 close_back_cusum
                        row["dm_str"],  # 12
                        row["decoder_time_idx"],
                        row["y_hat_cum_max"],
                        row["y_hat_cum_min"],
                        row["pred_img"],
                        row["error_cum_max"],
                        row["error_cum_min"],
                        row["rmse"],
                        row["mae"],
                        row["last_position"],
                        row["new_position"],
                        row["delta_position"],
                        row["px"],
                        row["pnl_delta"],
                    )
                #logging.info(f"new_position:{position}, prediction:{prediction}")
            logging.info(f"eod {test_date}")
            gc.collect()
        data_artifact.add(data_table, "trading_data")
        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        pnl_df = trader.pnl_df
        logging.info(f"pnl:{pnl_df}")
        if not pnl_df.empty:
            stats = self.compute_stats(pnl_df.pnl)
            for key, val in stats.items():
                wandb.run.summary[key] = val
            logging.info(f"stats:{stats}")

    def compute_stats(self, srs : pd.DataFrame, metric_suffix = ""):
        return {
            f"annual_return{metric_suffix}": annual_return(srs),
            f"annual_volatility{metric_suffix}": annual_volatility(srs),
            f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
            f"downside_risk{metric_suffix}": downside_risk(srs),
            f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
            f"max_drawdown{metric_suffix}": -max_drawdown(srs),
            f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
            f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
            f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
            / np.mean(np.abs(srs[srs < 0.0])),
        }

    def eval_model(self):
        # self.data_module = nhits.get_data_module(self.config)
        # calcualte metric by which to display
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        trainer_kwargs = {"logger": wandb_logger}
        logging.info(f"rows:{len(self.data_module.eval_data)}")
        data_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type="evaluation")
        metrics = SMAPE(reduction="none").to(self.device)
        data_table = viz_utils.create_example_viz_table(
            self.model.to(self.device),
            self.data_module.val_dataloader(),
            self.data_module.eval_data,
            metrics,
            self.config.job.eval_top_k,
        )
        data_artifact.add(data_table, "eval_data")

        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()
