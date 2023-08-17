import datetime
import gc
import logging
from typing import List

#import fiftyone as fo
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
import optuna

from pytorch_forecasting.metrics import (
    SMAPE,
    QuantileLoss,
)
from pytorch_forecasting.models.patch_tft_supervised.tuning import (
    optimize_hyperparameters,
)
import torch
import wandb

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr
from ats.model import model_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.search import faiss_builder
from ats.trading.trader import Trader

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

        self.ENCODER_LEN_MIN = 252  # 1 anio
        self.ENCODER_LEN_MAX = 252  # 1 anio
        self.DECODER_LEN = 63  # 1 trimestre
        self.TIME_IDX = "Time_idx"
        self.TARGET = ["Close", "Volume"]
        self.GROUP_ID = ["Ticker"]
        self.STATIC_CATEGORICALS = ["Ticker"]
        self.TIME_KNOW_CATEGORICALS = [
            "Date_day",
            "Date_month",
            "Date_day_week",
            "Date_day_year",
        ]
        self.TIME_KNOW_REALS = ["Time_idx"]
        self.TIME_UNKNOW_CATEGORICALS = []
        self.TIME_UNKNOW_REALS = ["Close", "Open", "Volume"]


class PatchTftSupervisedPipeline(Pipeline):
    def __init__(self, dataset="fut", config=None, run_id=None):
        super().__init__(config)
        self.dataset = dataset
        self.config = config
        self.env_mgr = EnvMgr(config, run_id)
        self.market_cal = self.env_mgr.market_cal
        self.heads = self.env_mgr.heads
        self.targets = self.env_mgr.targets
        self.md_mgr = market_data_mgr.MarketDataMgr(self.env_mgr)
        self.data_module = self.md_mgr.data_module()
        self.run_id = run_id

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
        opt = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            study_name=study_name,
        )
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
            trainer_kwargs=dict(
                limit_train_batches=30, accelerator="gpu", devices=-1, callbacks=[]
            ),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            study=opt,
            timeout=60 * 60 * 2,  # 2 horas
            **kwargs,
        )
        print("best_trial", study.best_trial.params)

    def test_model(self):
        test_dates = self.env_mgr.market_cal.valid_days(
            start_date=self.env_mgr.test_start_date, end_date=self.env_mgr.test_end_date
        )
        train_dataset = self.data_module.training
        train_data = self.data_module.train_data
        future_data = self.data_module.test_data
        train_data = train_data[train_data.ticker.isin(["ES"])]
        future_data = future_data[future_data.ticker.isin(["ES"])]

        # logging.info(f"train_data:{train_data.iloc[-2:]}")
        # logging.info(f"future_data:{future_data.iloc[:2]}")
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

        trader = Trader(
            self.md_mgr,
            self.model,
            wandb_logger,
            target_size,
            future_data,
            train_data,
            train_dataset,
            self.config,
            self.market_cal,
        )
        for idx in range(len(test_dates)-2):
            test_date = test_dates[idx]
            schedule = self.market_cal.schedule(
                start_date=test_date, end_date=test_date
            )
            time_range = mcal.date_range(schedule, frequency=f"{self.config.job.time_interval_minutes}min")
            logging.info(f"sod {test_date}, schedule:{time_range}")
            for utc_time in time_range:
                row = trader.on_interval(utc_time)
                #logging.info(f"got row:{row}")
                if row:
                    #logging.info(f"row:{row}")
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
                # logging.info(f"new_position:{position}, prediction:{prediction}")
                gc.collect()
            logging.info(f"eod {test_date}")
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

    def compute_stats(self, srs: pd.DataFrame, metric_suffix=""):
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

    def build_search(self):
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        trainer_kwargs = {"logger": wandb_logger}
        logging.info(f"rows:{len(self.data_module.eval_data)}")
        search_builder = faiss_builder.FaissBuilder(self.env_mgr, self.model, self.md_mgr, wandb_logger)
        search_builder.build_embedding_cache_if_not_exists()

    def search_example(self):
        from vss.metrics.core import BestChoiceImagesDataset, MetricClient
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        trainer_kwargs = {"logger": wandb_logger}
        logging.info(f"rows:{len(self.data_module.eval_data)}")
        #dataset = fo.Dataset()
        full_data = self.data_module.full_data
        full_data = full_data[full_data.ticker.isin(["ES"])]
        
        example_df = pd.read_csv(self.config.job.example_key_file, header=None, columns=["ticker","timestamp","time"])
        metric_client = MetricClient(cfg=self.cfg)
        for idx, row in example_df.iterrows():
            results = client.search_by_ticker_time_idx(row["ticker"], float(row["timestamp"]), "futures")
            logging.error(f"results:{results}")
