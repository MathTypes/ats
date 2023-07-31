import logging

import pandas as pd

from hydra import initialize, compose
import math
import pandas_market_calendars as mcal
import torch

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr
from ats.model import model_utils
from ats.trading.trader import Trader


def test_create_trader():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(config_name="test", overrides=[], return_hydra_config=True)
        env_mgr = EnvMgr(cfg)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        market_cal = md_mgr.market_cal
        wandb_logger = None
        data_module = md_mgr.data_module()
        model = model_utils.get_patch_tft_supervised_model(
            cfg, data_module, env_mgr.heads
        )
        train_dataset = data_module.training
        train_data = data_module.train_data
        future_data = data_module.test_data

        trader = Trader(
            md_mgr,
            model,
            wandb_logger,
            env_mgr.target_size,
            future_data,
            train_data,
            train_dataset,
            cfg,
            market_cal,
        )
        first_train_data_row = train_data.iloc[0]
        # 2008-02-07 08:30:00 - we need 201 prior intervals. Each day
        # has about 40 intervals. It is strange why we go to 2008-02-07 instead
        # of 2008-01-10.
        assert first_train_data_row["timestamp"] == 1243807200


def assert_map_close(test_map, expected_map):
    for key, val in expected_map.items():
        if isinstance(val, (int, float)):
            assert math.isclose(
                test_map[key], val, rel_tol=0.15
            ), f"{key} should be close"
        else:
            assert test_map[key] == expected_map[key], f"{key} should match"


def test_on_interval():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(config_name="test", overrides=[], return_hydra_config=True)
        env_mgr = EnvMgr(cfg)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        market_cal = md_mgr.market_cal
        wandb_logger = None
        data_module = md_mgr.data_module()
        model = model_utils.get_patch_tft_supervised_model(
            cfg, data_module, env_mgr.heads
        )
        train_dataset = data_module.training
        train_data = data_module.train_data
        future_data = data_module.test_data

        trader = Trader(
            md_mgr,
            model,
            wandb_logger,
            env_mgr.target_size,
            future_data,
            train_data,
            train_dataset,
            cfg,
            market_cal,
        )
        first_train_data_row = train_data.iloc[0]
        logging.info(f"first_train_data_row:{first_train_data_row}")
        last_train_data_row = train_data.iloc[-1]
        logging.info(f"last_train_data_row:{last_train_data_row}")
        test_date = cfg.job.test_start_date
        schedule = market_cal.schedule(start_date=test_date, end_date=test_date)
        time_range = mcal.date_range(
            schedule, frequency=f"{cfg.job.time_interval_minutes}min"
        )
        logging.info(f"sod {test_date}, schedule:{time_range}")
        utc_time1 = time_range[0]
        row1 = trader.on_interval(utc_time1)
        expected_row1 = {
            "ticker": "ES",
            "time_idx": 26922,
            "day_of_week": 2,
            "hour_of_day": 19,
            "year": 2010,
            "month": 6,
            "day_of_month": 30,
            "y_close_cum_max": 0.0024,
            "y_close_cum_min": 0.0011,
            "close_back_cumsum": 0,
            "close_back": 0.001085383608724122,
            "dm_str": "20100630-190000",
            "decoder_time_idx": 26922,
            "y_hat_cum_max": -0.0002,
            "y_hat_cum_min": -0.0016,
            "error_cum_max": -0.0025,
            "error_cum_min": -0.0026,
            "rmse": 0,
            "mae": 0,
            "last_position": 0,
            "new_position": -2.0,
            "delta_position": -2.0,
            "px": 881.25,
            "pnl_delta": 0.0,
        }
        assert_map_close(row1, expected_row1)
        expected_pnl1 = {
            "ticker": "ES",
            "timestamp": 1277937000,
            "px": 881.25,
            "last_px": 881.25,
            "pos": -2.0,
            "pnl": 0.0,
        }
        assert_map_close(trader.pnl_df.iloc[-1], expected_pnl1)

        expected_data_row1 = {}
        logging.error(f"trader.current_data_row:{trader.current_data_row}")
        assert_map_close(trader.current_data_row, expected_data_row1)
        
        utc_time2 = time_range[1]
        row2 = trader.on_interval(utc_time2)
        expected_row2 = {
            "ticker": "ES",
            "time_idx": 26923,
            "day_of_week": 2,
            "hour_of_day": 19,
            "year": 2010,
            "month": 6,
            "day_of_month": 30,
            "y_close_cum_max": 0.001264793798327446,
            "y_close_cum_min": -0.004711860790848732,
            "close_back_cumsum": 0,
            "close_back": 0.0009035873028278019,
            "dm_str": "20100630-193000",
            "decoder_time_idx": 26923,
            "y_hat_cum_max": -0.00024568778462707996,
            "y_hat_cum_min": -0.0016140901716426015,
            "error_cum_max": -0.0015,
            "error_cum_min": 0.0031,
            "rmse": 0,
            "mae": 0,
            "last_position": -2.0,
            "new_position": -2.0,
            "delta_position": 0.0,
            "px": 882.75,
            "pnl_delta": -3.0,
        }
        assert_map_close(row2, expected_row2)
        expected_pnl2 = {
            "ticker": "ES",
            "timestamp": 1277938800,
            "px": 882.75,
            "last_px": 881.25,
            "pos": -2.0,
            "pnl": -3.0,
        }
        assert_map_close(trader.pnl_df.iloc[-1], expected_pnl2)

        expected_data_row2 = {}
        logging.error(f"trader.current_data_row:{trader.current_data_row}")
        assert_map_close(trader.current_data_row, expected_data_row2)

