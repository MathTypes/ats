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
        cfg = compose(config_name="test",
                      overrides=[
                          "job.test_start_date=2010-07-30",
                          "job.test_end_date=2010-07-30",
                      ],
                      return_hydra_config=True)
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

        expected_data_row1 = {
            "open": 881.5,
            "high": 882.0,
            "low": 881.0,
            "close": 881.25,
            "volume": 1312,
            "dv": 1156529.75,
            "ticker": "ES",
            "month": 6,
            "year": 2010,
            "hour_of_day": 18,
            "day_of_week": 2,
            "day_of_month": 30,
            "timestamp": 1277937000,
            "time_idx": 26921,
            "cum_volume": 1255864462,
            "cum_dv": 1116586590539.0,
            "close_back": 0.0,
            "volume_back": -1.307551,
            "dv_back": -1.308662,
            "close_fwd": 0.0,
            "volume_fwd": 0.305061,
            "dv_fwd": 0.305754,
            "close_back_cumsum": -0.191181,
            "volume_back_cumsum": 0.105868,
            "close_high_5_ff": -0.143814,
            "time_high_5_ff": 1277103600.0,
            "close_low_5_ff": -0.191181,
            "time_low_5_ff": 1277937000.0,
            "close_high_11_ff": -0.117914,
            "time_high_11_ff": 1277103600.0,
            "close_low_11_ff": -0.191181,
            "time_low_11_ff": 1277937000.0,
            "close_high_21_ff": -0.117914,
            "time_high_21_ff": 1277103600.0,
            "close_low_21_ff": -0.191181,
            "time_low_21_ff": 1277937000.0,
            "close_high_51_ff": -0.063732,
            "time_high_51_ff": 1272265200.0,
            "close_low_51_ff": -0.191181,
            "time_low_51_ff": 1277937000.0,
            "close_high_201_ff": -0.063732,
            "time_high_201_ff": 1272265200.0,
            "close_low_201_ff": -0.191181,
            "time_low_201_ff": 1277937000.0,
            "rsi": 25.915513,
            "macd": -4.270789,
            "macd_signal": -2.73349,
            "bb_high": 905.265079,
            "bb_low": 878.359921,
            "sma_50": 894.65,
            "sma_200": 918.18375,
            "daily_returns": 0.0,
            "daily_vol": 0.002577,
            "macd_8_24": -3.330439,
            "macd_16_48": -5.692596,
            "macd_32_96": -4.804764,
            "week_of_year": 26,
            "month_of_year": 6,
            "weekly_close_time": 1278104400,
            "monthly_close_time": 1277931600,
            "option_expiration_time": 1279314000,
            "new_york_open_time": 1277848800,
            "new_york_close_time": 1277931600,
            "london_open_time": 1277881200,
            "london_close_time": 1277911800,
            "time_to_new_york_open": -88200,
            "time_to_new_york_close": -5400,
            "time_to_london_open": -55800,
            "time_to_london_close": -25200,
            "time_to_weekly_close": 167400,
            "time_to_monthly_close": -5400,
            "time_to_option_expiration": 1377000,
            "time_to_high_5_ff": -833400.0,
            "time_to_low_5_ff": 0.0,
            "time_to_high_11_ff": -833400.0,
            "time_to_low_11_ff": 0.0,
            "time_to_high_21_ff": -833400.0,
            "time_to_low_21_ff": 0.0,
            "time_to_high_51_ff": -5671800.0,
            "time_to_low_51_ff": 0.0,
            "time_to_high_201_ff": -5671800.0,
            "time_to_low_201_ff": 0.0,
            "relative_time_idx": 0.0,
        }
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

        expected_data_row2 = {
            "open": 881.25,
            "high": 882.75,
            "low": 881.25,
            "close": 882.75,
            "volume": 1780,
            "dv": 1570161.25,
            "ticker": "ES",
            "month": 6,
            "year": 2010,
            "hour_of_day": 19,
            "day_of_week": 2,
            "day_of_month": 30,
            "timestamp": 1277938800,
            "time_idx": 26922,
            "cum_volume": 1255866242,
            "cum_dv": 1116588160700.25,
            "close_back": 0.001085,
            "volume_back": 0.30466,
            "dv_back": 0.305754,
            "close_fwd": -0.001701,
            "volume_fwd": 0.761278,
            "close_back_cumsum": -0.190096,
            "volume_back_cumsum": 0.410528,
            "close_high_5_ff": -0.143814,
            "time_high_5_ff": 1277103600.0,
            "close_low_5_ff": -0.191181,
            "time_low_5_ff": 1277937000.0,
            "close_high_11_ff": -0.117914,
            "time_high_11_ff": 1277103600.0,
            "close_low_11_ff": -0.191181,
            "time_low_11_ff": 1277937000.0,
            "close_high_21_ff": -0.117914,
            "time_high_21_ff": 1277103600.0,
            "close_low_21_ff": -0.191181,
            "time_low_21_ff": 1277937000.0,
            "close_high_51_ff": -0.063732,
            "time_high_51_ff": 1272265200.0,
            "close_low_51_ff": -0.191181,
            "time_low_51_ff": 1277937000.0,
            "close_high_201_ff": -0.063732,
            "time_high_201_ff": 1272265200.0,
            "close_low_201_ff": -0.191181,
            "time_low_201_ff": 1277937000.0,
            "rsi": 30.783473,
            "macd": -4.297539,
            "macd_signal": -3.0463,
            "bb_high": 905.302321,
            "bb_low": 877.297679,
            "sma_50": 894.475,
            "sma_200": 917.94625,
            "daily_returns": 0.001702,
            "daily_vol": 0.002568,
            "macd_8_24": -3.299276,
            "macd_16_48": -5.561259,
            "macd_32_96": -4.699088,
            "week_of_year": 26,
            "month_of_year": 6,
            "weekly_close_time": 1278104400,
            "monthly_close_time": 1277931600,
            "option_expiration_time": 1279314000,
            "new_york_open_time": 1277848800,
            "new_york_close_time": 1277931600,
            "london_open_time": 1277881200,
            "london_close_time": 1277911800,
            "time_to_new_york_open": -90000,
            "time_to_new_york_close": -7200,
            "time_to_london_open": -57600,
            "time_to_london_close": -27000,
            "time_to_weekly_close": 165600,
            "time_to_monthly_close": -7200,
            "time_to_option_expiration": 1375200,
            "time_to_high_5_ff": -835200.0,
            "time_to_low_5_ff": -1800.0,
            "time_to_high_11_ff": -835200.0,
            "time_to_low_11_ff": -1800.0,
            "time_to_high_21_ff": -835200.0,
            "time_to_low_21_ff": -1800.0,
            "time_to_high_51_ff": -5673600.0,
            "time_to_low_51_ff": -1800.0,
            "time_to_high_201_ff": -5673600.0,
            "time_to_low_201_ff": -1800.0,
            "relative_time_idx": 0.0,
        }
        logging.error(f"trader.current_data_row:{trader.current_data_row}")
        assert_map_close(trader.current_data_row, expected_data_row2)


def test_on_interval_no_future():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
                "job.test_start_date=2010-07-30",
                "job.test_end_date=2010-07-30",
            ],
            return_hydra_config=True,
        )
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
        utc_time1 = time_range[-3]
        row1 = trader.on_interval(utc_time1)
        logging.error(f"row1:{row1}")
        expected_row1 = {
            "ticker": "ES",
            "time_idx": 16029,
            "day_of_week": 4.0,
            "hour_of_day": 17.0,
            "year": 2010.0,
            "month": 7.0,
            "day_of_month": 30.0,
            "y_close_cum_max": -0.22008858621120453,
            "y_close_cum_min": -0.4497380554676056,
            "close_back_cumsum": 0,
            "close_back": -0.22008858436751844,
            "dm_str": "20100730-170000",
            "decoder_time_idx": 16029,
            "y_hat_cum_max": 0.001522403908893466,
            "y_hat_cum_min": -0.0005837632925249636,
            "error_cum_max": 0.2216,
            "error_cum_min": 0.4492,
            "rmse": 0,
            "mae": 0,
            "last_position": 0,
            "new_position": -0.0,
            "delta_position": -0.0,
            "px": 958.0,
            "pnl_delta": 0.0,
        }
        assert_map_close(row1, expected_row1)
        expected_pnl1 = {
            "ticker": "ES",
            "px": 958.0,
            "last_px": 954.75,
            "pos": -0.0,
            "pnl": 0.0,
        }
        logging.error(f"actual_pnl1:{trader.pnl_df.iloc[-1]}")
        assert_map_close(trader.pnl_df.iloc[-1], expected_pnl1)

        expected_data_row1 = {
            "open": 955.5,
            "high": 958.25,
            "low": 954.75,
            "close": 958.0,
            "volume": 101481,
            "dv": 97116725.25,
            "ticker": "ES",
            "month": 7.0,
            "year": 2010.0,
            "hour_of_day": 16.0,
            "day_of_week": 4.0,
            "day_of_month": 30.0,
            "timestamp": 1280520000.0,
            "time_idx": 16028,
            "cum_volume": 695461164.0,
            "cum_dv": 617195631139.0,
            "close_back": 0.0015444018513743885,
            "volume_back": -0.8110188970884149,
            "dv_back": -0.8118538513872089,
            "close_back_cumsum": 0.23773856836536744,
            "volume_back_cumsum": 4.106469047156228,
            "close_rolling_5d_max": 0.24899188405209483,
            "close_high_5_ff": 0.24899188405209483,
            "close_high_5_bf": 0.24899188405209483,
            "time_high_5_ff": 1280235600.0,
            "close_rolling_5d_min": 0.22774365489975334,
            "close_low_5_ff": 0.22774365489975334,
            "close_low_5_bf": 0.22774365489975334,
            "time_low_5_ff": 1280493000.0,
            "close_rolling_11d_max": 0.24899188405209483,
            "close_high_11_ff": 0.24899188405209483,
            "close_high_11_bf": 0.24899188405209483,
            "time_high_11_ff": 1280235600.0,
            "close_rolling_11d_min": 0.2037305827584479,
            "close_low_11_ff": 0.2037305827584479,
            "close_low_11_bf": 0.2037305827584479,
            "time_low_11_ff": 1278376200.0,
            "close_rolling_21d_max": 0.24899188405209483,
            "close_high_21_ff": 0.24899188405209483,
            "close_high_21_bf": 0.24899188405209483,
            "time_high_21_ff": 1280235600.0,
            "close_rolling_21d_min": 0.16962714250622124,
            "close_low_21_ff": 0.16962714250622124,
            "close_low_21_bf": 0.16962714250622124,
            "time_low_21_ff": 1278376200.0,
            "close_rolling_51d_max": 0.25692910380175515,
            "close_high_51_ff": 0.25692910380175515,
            "close_high_51_bf": 0.25692910380175515,
            "time_high_51_ff": 1272265200.0,
            "close_rolling_51d_min": 0.16962714250622124,
            "close_low_51_ff": 0.16962714250622124,
            "close_low_51_bf": 0.16962714250622124,
            "time_low_51_ff": 1278376200.0,
            "close_rolling_201d_max": 0.31111140852979613,
            "close_high_201_ff": 0.31111140852979613,
            "close_high_201_bf": 0.31111140852979613,
            "time_high_201_ff": 1272265200.0,
            "close_rolling_201d_min": 0.16962714250622124,
            "close_low_201_ff": 0.16962714250622124,
            "close_low_201_bf": 0.16962714250622124,
            "time_low_201_ff": 1278376200.0,
            "rsi": 57.514945421452815,
            "macd": 1.2978311035967636,
            "macd_signal": 0.5510021683449411,
            "bb_high": 960.4832145273556,
            "bb_low": 944.1167854726443,
            "sma_50": 952.22,
            "sma_200": 962.48625,
            "daily_returns": 0.002354172116139086,
            "daily_vol": 0.002773185600211478,
            "macd_8_24": 0.14542452910511552,
            "macd_16_48": -0.6068306985564004,
            "macd_32_96": -0.43902525079671023,
            "week_of_year": 30.0,
            "month_of_year": 7.0,
            "weekly_close_time": 1280523600.0,
            "monthly_close_time": 1280523600.0,
            "option_expiration_time": 1282338000.0,
            "new_york_open_time": 1280496600.0,
            "new_york_last_open_time": 1280496600.0,
            "new_york_open_cum_dv": 615781665351.0,
            "new_york_open_cum_volume": 693980357.0,
            "vwap_since_new_york_open": 0.0021548360904661834,
            "vwap_around_new_york_open": 952.09421447311,
            "ret_from_vwap_around_new_york_open": 0.004058833288103969,
            "new_york_close_time": 1280520000.0,
            "london_open_time": 1280473200.0,
            "london_last_open_time": 1280473200.0,
            "london_open_cum_dv": 615233801371.25,
            "london_open_cum_volume": 693401642.0,
            "vwap_since_london_open": 0.003734275354843497,
            "vwap_around_london_open": 951.9928899270551,
            "ret_from_vwap_around_london_open": 0.004128613940303616,
            "london_close_time": 1280503800.0,
            "time_to_new_york_open": -23400.0,
            "time_to_new_york_last_open": -23400.0,
            "time_to_new_york_close": 0.0,
            "time_to_london_open": -46800.0,
            "time_to_london_last_open": -46800.0,
            "time_to_london_close": -16200.0,
            "time_to_weekly_close": 3600.0,
            "time_to_monthly_close": 3600.0,
            "time_to_option_expiration": 1818000.0,
            "time_to_high_5_ff": -284400.0,
            "time_to_low_5_ff": -27000.0,
            "time_to_high_11_ff": -284400.0,
            "time_to_low_11_ff": -2143800.0,
            "time_to_high_21_ff": -284400.0,
            "time_to_low_21_ff": -2143800.0,
            "time_to_high_51_ff": -8254800.0,
            "time_to_low_51_ff": -2143800.0,
            "time_to_high_201_ff": -8254800.0,
            "time_to_low_201_ff": -2143800.0,
            "relative_time_idx": 0.0,
        }
        logging.error(f"trader.current_data_row:{trader.current_data_row.to_dict()}")
        assert_map_close(trader.current_data_row, expected_data_row1)
