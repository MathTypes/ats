import math
import logging
import unittest.mock as mock

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

import pandas_market_calendars as mcal

from hydra import initialize, compose
from omegaconf import OmegaConf

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr
from ats.model import model_utils
from ats.trading.trader import Trader
from ats.util import logging_utils


# THe config path is relative to the file calling initialize (this file)
def test_on_interval():
    pytest_plugins = ['pytest_profiling']
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(config_name="test_dev", overrides=["job.train_start_date=2008-01-01",
                                                         "job.eval_start_date=2008-05-01",
                                                         "job.eval_end_date=2008-06-01",
                                                         "job.test_start_date=2008-05-01",
                                                         "job.eval_end_date=2008-06-01"])
        env_mgr = EnvMgr(cfg)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        market_cal = md_mgr.market_cal
        #wandb_logger = WandbLogger(project="ATS", log_model=True)
        wandb_logger = None
        data_module = model_utils.get_data_module(env_mgr, True)    
        model = model_utils.get_patch_tft_supervised_model(
            cfg, data_module, env_mgr.heads)
        train_dataset = data_module.training
        train_data = data_module.train_data
        future_data = data_module.test_data

        trader = Trader(md_mgr, model, wandb_logger,
                        env_mgr.target_size, future_data,
                        train_data, train_dataset, cfg, market_cal)
        first_train_data_row = train_data.iloc[0]
        # 2008-02-07 08:30:00 - we need 201 prior intervals. Each day
        # has about 40 intervals. It is strange why we go to 2008-02-07 instead
        # of 2008-01-10.
        assert first_train_data_row["timestamp"] == 1202391000
        logging.error(f"train_data start:{train_data[:50]['time']}")
        logging.error(f"train_data end:{train_data[-50:]['time']}")
        logging.error(f"future_data start:{future_data[:2]}")
        logging.error(f"future_data end:{future_data[-2:]}")
