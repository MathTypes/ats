import math
import logging
import unittest.mock as mock

import hydra
import numpy as np
from omegaconf import DictConfig
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
