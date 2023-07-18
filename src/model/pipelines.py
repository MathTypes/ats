from collections import defaultdict
import datetime
from dateutil import parser
from io import BytesIO
import logging

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
import pytz
import plotly.graph_objects as go
import PIL
from plotly.subplots import make_subplots
from timeseries_transformer import TimeSeriesTFT
import torch
import wandb

from data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from models import AttentionEmbeddingLSTM
import model_utils
from prediction import prediction_utils
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
        self.init_env()

    def init_env(self):
        self.train_start_date = datetime.datetime.strptime(
            self.config.job.train_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.eval_start_date = datetime.datetime.strptime(
            self.config.job.eval_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.eval_end_date = datetime.datetime.strptime(
            self.config.job.eval_end_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.test_start_date = datetime.datetime.strptime(
            self.config.job.test_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.test_end_date = datetime.datetime.strptime(
            self.config.job.test_end_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.max_lags = self.config.job.max_lags
        if self.config.job.mode == "train":
            self.train_start_date = datetime.datetime.strptime(
                self.config.job.train_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
        elif self.config.job.mode == "test":
            self.data_start_date = self.test_start_date - datetime.timedelta(
                days=self.max_lags
            )
            self.market_cal = mcal.get_calendar(self.config.job.market)
        logging.info(f"train_start_date:{self.train_start_date}, test_start_date:{self.test_start_date}, test_end_date:{self.test_end_date}")
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        logging.info(f"head:{self.heads}, targets:{self.targets}")
        start_date = self.train_start_date
        if not start_date:
            start_date = self.data_start_date
        logging.info(
            f"start_date:{start_date}, test_start_date:{self.test_start_date}, test_end_date:{self.test_end_date}"
        )
        self.data_module = model_utils.get_data_module(
            self.config,
            self.config.dataset.base_dir,
            start_date,
            self.eval_start_date,
            self.eval_end_date,
            self.test_start_date,
            self.test_end_date,
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
            start_date=self.test_start_date, end_date=self.test_end_date
        )
        train_dataset = self.data_module.training
        train_data = self.data_module.train_data
        future_data = self.data_module.test_data

        logging.info(f"train_data:{train_data.iloc[-2:]}")
        logging.info(f"future_data:{future_data.iloc[:2]}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        last_time_idx = train_data.iloc[-1]["time_idx"]
        last_data_time = None
        position_map = {}
        prediction_map = {}
        last_position_map = {} 
        last_px_map = {}
        last_data = train_data.iloc[-1]
        logging.info(f"last_data:{last_data}")
        last_px_map[last_data.ticker] = last_data.close
        logging.info(f"initial_last_px_map:{last_px_map}")
        last_position_map = defaultdict(lambda:0,last_position_map)
        pnl_df = pd.DataFrame(columns = ["ticker","timestamp","px","last_px","pos","pnl"])
        for test_date in test_dates:
            schedule = self.market_cal.schedule(
                start_date=test_date, end_date=test_date
            )
            logging.info(f"sod {test_date}")
            time_range = mcal.date_range(schedule, frequency="30M")
            max_prediction_length = self.config.model.prediction_length
            for utc_time in time_range:
                nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
                #logging.info(f"looking up nyc_time:{nyc_time}")
                # 1. prepare prediction with latest prices
                # 2. run inference to get returns and new positions
                # 3. update PNL and positions
                new_data = future_data[
                    (future_data.timestamp == nyc_time.timestamp())
                    & (future_data.ticker == "ES")
                ]
                if new_data.empty:
                    if last_data_time is None or nyc_time < last_data_time + datetime.timedelta(minutes=self.config.dataset.max_stale_minutes):
                        continue
                    else:
                        logging.info(f"data is too stale, now:{nyc_time}, last_data_time:{last_data_time}")
                        #exit(0)
                        continue
                last_data_time = nyc_time
                last_time_idx += 1
                new_data["time_idx"] = last_time_idx
                #logging.info(f"running step {nyc_time}, new_data:{new_data}")
                train_dataset.add_new_data(new_data, self.config.job.time_interval_minutes)
                #logging.info(f"new_train_dataset:{train_dataset.raw_data[-3:]}, last_time_idex={last_time_idx}")
                new_prediction_data = train_dataset.filter(
                    lambda x: (x.time_idx_last == last_time_idx)
                )
                # new_prediction_data is the last encoder_data, we need to add decoder_data based on
                # known features or lagged unknown features
                prediction, y_quantiles = prediciton_utils.predict(self.model, new_prediction_data, wandb_logger)
                #prediction, position = y_hats
                logging.info(f"prediction:{prediction}")
                #y_quantiles = to_list(self.model.to_quantiles(new_raw_predictions.output,
                #                                              **quantiles_kwargs))[0]
                logging.info(f"y_quantiles:{y_quantiles}")
                #logging.info(f"y_hats:{y_hats}")
                #logging.info(f"y:{new_raw_predictions.y}")
                new_data_row = new_data.iloc[0]
                ticker = new_data_row.ticker
                px = new_data_row.close
                last_position = last_position_map[ticker]
                last_px = last_px_map[ticker]
                pnl_delta = last_position * (px - last_px)
                new_position = position[0][0].item()
                df2 = {'ticker': ticker, 'timestamp': new_data_row.timestamp,
                       'px': px, 'last_px' : last_px,
                       'pos': new_position,
                       'pnl':pnl_delta}
                logging.info(f"new_df:{df2}")
                pnl_df = pnl_df.append(df2, ignore_index = True)
                last_position_map[ticker] = new_position
                last_px_map[ticker] = px

                logging.info(f"last_position_map:{last_position_map}")
                logging.info(f"last_px_map:{last_px_map}")
                #logging.info(f"new_position:{position}, prediction:{prediction}")
            logging.info(f"eod {test_date}")
        stats = self.compute_stats(pnl_df.pnl)
        logging.info(f"pnl:{pnl_df}")
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
