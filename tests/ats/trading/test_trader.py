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
            'ticker': 'ES',
            'time_idx': 15986, 'day_of_week': 3, 'hour_of_day': 19, 'year': 2010, 'month': 7, 'day_of_month': 29,
            'y_close_cum_max': 0.00017189513891935349, 'y_close_cum_min': -0.0020650499500334263, 'close_back_cumsum': 0, 'close_back': 0.00017189514438609166, 'dm_str': '20100729-190000', 'decoder_time_idx': 15986, 'y_hat_cum_max': -4.350992094259709e-05, 'y_hat_cum_min': -0.0013000972103327513,
            'error_cum_max': -0.0002, 'error_cum_min': 0.0008, 'rmse': 0, 'mae': 0, 'last_position': 0, 'new_position': -0.0, 'delta_position': -0.0, 'px': 954.25, 'pnl_delta': -0.0
        }
        logging.error(f"row1:{row1}")
        assert_map_close(row1, expected_row1)
        expected_pnl1 = {
            "ticker": "ES",
            "px": 954.25,
            "last_px": 954.75,
            "pos": -0.0,
            "pnl": 0.0,
        }
        logging.error(f"pnl:{trader.pnl_df.iloc[-1]}")
        assert_map_close(trader.pnl_df.iloc[-1], expected_pnl1)

        expected_data_row1 = {
            'open': 954.75, 'high': 954.75, 'low': 954.0, 'close': 954.25, 'volume': 906, 'dv': 864617.25, 'ticker': 'ES',
            'month': 7, 'year': 2010, 'hour_of_day': 18, 'day_of_week': 3, 'day_of_month': 29,
            'timestamp': 1280442600, 'time_idx': 15985, 'cum_volume': 693339230, 'cum_dv': 615174375044.75, 'close_back': -0.0003437607459089165, 'volume_back': -0.5783195750763417, 'dv_back': -0.5796911359943646, 'close_back_cumsum': 0.2351632385876341, 'volume_back_cumsum': -0.6099331499941005, 'close_rolling_5d_max': 0.24899188405209483, 'close_high_5_ff': 0.24899188405209483, 'close_high_5_bf': 0.24899188405209483, 'time_high_5_ff': 1280235600.0, 'close_rolling_5d_min': 0.22722394938783896, 'close_low_5_ff': 0.22722394938783896, 'close_low_5_bf': 0.22722394938783896, 'time_low_5_ff': 1279627200.0, 'close_rolling_11d_max': 0.24899188405209483, 'close_high_11_ff': 0.24899188405209483, 'close_high_11_bf': 0.24899188405209483, 'time_high_11_ff': 1280235600.0, 'close_rolling_11d_min': 0.2037305827584479, 'close_low_11_ff': 0.2037305827584479, 'close_low_11_bf': 0.2037305827584479, 'time_low_11_ff': 1278376200.0, 'close_rolling_21d_max': 0.24899188405209483, 'close_high_21_ff': 0.24899188405209483, 'close_high_21_bf': 0.24899188405209483, 'time_high_21_ff': 1280235600.0, 'close_rolling_21d_min': 0.16962714250622124, 'close_low_21_ff': 0.16962714250622124, 'close_low_21_bf': 0.16962714250622124, 'time_low_21_ff': 1278376200.0, 'close_rolling_51d_max': 0.25692910380175515, 'close_high_51_ff': 0.25692910380175515, 'close_high_51_bf': 0.25692910380175515, 'time_high_51_ff': 1272265200.0, 'close_rolling_51d_min': 0.16962714250622124, 'close_low_51_ff': 0.16962714250622124, 'close_low_51_bf': 0.16962714250622124, 'time_low_51_ff': 1278376200.0, 'close_rolling_201d_max': 0.31111140852979613, 'close_high_201_ff': 0.31111140852979613, 'close_high_201_bf': 0.31111140852979613, 'time_high_201_ff': 1272265200.0, 'close_rolling_201d_min': 0.16962714250622124, 'close_low_201_ff': 0.16962714250622124, 'close_low_201_bf': 0.16962714250622124, 'time_low_201_ff': 1278376200.0, 'rsi': 39.96373595570956, 'macd': -2.100849865669602, 'macd_signal': -1.997282338412578, 'bb_high': 968.408358856576, 'bb_low': 946.641641143424, 'sma_50': 960.63, 'sma_200': 963.7575, 'daily_returns': -0.0005236973029588698, 'daily_vol': 0.0022435084745550966, 'macd_8_24': -1.446092200034012, 'macd_16_48': -1.0246485076304066, 'macd_32_96': 0.15132249796714028, 'week_of_year': 30, 'month_of_year': 7, 'weekly_close_time': 1280523600, 'monthly_close_time': 1280523600, 'option_expiration_time': 1282338000, 'new_york_open_time': 1280410200, 'new_york_last_open_time': 1280410200.0, 'new_york_open_cum_dv': 613424553295.0, 'new_york_open_cum_volume': 691510058.0, 'vwap_since_new_york_open': -0.0016280909260393273, 'vwap_around_new_york_open': 964.3921794675732, 'ret_from_vwap_around_new_york_open': -0.006949958006351231, 'new_york_close_time': 1280433600, 'london_open_time': 1280386800, 'london_last_open_time': 1280386800.0, 'london_open_cum_dv': 613070157512.25, 'london_open_cum_volume': 691143683.0, 'vwap_since_london_open': -0.002851291953712476, 'vwap_around_london_open': 963.5447276863375, 'ret_from_vwap_around_london_open': -0.0063710850102634, 'london_close_time': 1280417400, 'time_to_new_york_open': -32400, 'time_to_new_york_last_open': -32400.0, 'time_to_new_york_close': -9000, 'time_to_london_open': -55800, 'time_to_london_last_open': -55800.0, 'time_to_london_close': -25200, 'time_to_weekly_close': 81000, 'time_to_monthly_close': 81000, 'time_to_option_expiration': 1895400, 'time_to_high_5_ff': -207000.0, 'time_to_low_5_ff': -815400.0, 'time_to_high_11_ff': -207000.0, 'time_to_low_11_ff': -2066400.0, 'time_to_high_21_ff': -207000.0, 'time_to_low_21_ff': -2066400.0, 'time_to_high_51_ff': -8177400.0, 'time_to_low_51_ff': -2066400.0, 'time_to_high_201_ff': -8177400.0, 'time_to_low_201_ff': -2066400.0, 'relative_time_idx': 0.0
        }
        logging.error(f"trader.current_data_row:{trader.current_data_row.to_dict()}")
        assert_map_close(trader.current_data_row, expected_data_row1)

        utc_time2 = time_range[1]
        row2 = trader.on_interval(utc_time2)
        expected_row2 = {
            'ticker': 'ES',
            'time_idx': 15987, 'day_of_week': 3, 'hour_of_day': 19, 'year': 2010, 'month': 7, 'day_of_month': 29,
            'y_close_cum_max': -0.0005157741252332926, 'y_close_cum_min': -0.002409225096926093, 'close_back_cumsum': 0, 'close_back': -0.0005157741023822382, 'dm_str': '20100729-193000', 'decoder_time_idx': 15987, 'y_hat_cum_max': -8.111371425911784e-05, 'y_hat_cum_min': -0.0013088630512356758,
            'error_cum_max': 0.0004, 'error_cum_min': 0.0011, 'rmse': 0, 'mae': 0, 'last_position': -0.0, 'new_position': -0.0, 'delta_position': 0.0, 'px': 954.5, 'pnl_delta': -0.0
        }
        logging.error(f"row2:{row2}")
        assert_map_close(row2, expected_row2)
        expected_pnl2 = {
            "ticker": "ES",
            "px": 954.5,
            "last_px": 954.25,
            "pos": -0.0,
            "pnl": -0.0,
        }
        logging.error(f"trader.pnl_df.iloc[-1]:{trader.pnl_df.iloc[-1]}")
        assert_map_close(trader.pnl_df.iloc[-1], expected_pnl2)

        expected_data_row2 = {
            'open': 954.0, 'high': 954.75, 'low': 954.0, 'close': 954.5, 'volume': 1775, 'dv': 1693735.25, 'ticker': 'ES',
            'month': 7, 'year': 2010, 'hour_of_day': 19, 'day_of_week': 3, 'day_of_month': 29,
            'timestamp': 1280444400, 'time_idx': 15986, 'cum_volume': 693341005, 'cum_dv': 615176068780.0, 'close_back': 0.00017189514438609166, 'volume_back': 0.6714374495533582, 'dv_back': 0.6724046526458789, 'close_back_cumsum': 0.2353351337320202, 'volume_back_cumsum': 0.061504299559257625, 'close_rolling_5d_max': 0.24899188405209483, 'close_high_5_ff': 0.24899188405209483, 'close_high_5_bf': 0.24899188405209483, 'time_high_5_ff': 1280235600.0, 'close_rolling_5d_min': 0.22722394938783896, 'close_low_5_ff': 0.22722394938783896, 'close_low_5_bf': 0.22722394938783896, 'time_low_5_ff': 1279627200.0, 'close_rolling_11d_max': 0.24899188405209483, 'close_high_11_ff': 0.24899188405209483, 'close_high_11_bf': 0.24899188405209483, 'time_high_11_ff': 1280235600.0, 'close_rolling_11d_min': 0.2037305827584479, 'close_low_11_ff': 0.2037305827584479, 'close_low_11_bf': 0.2037305827584479, 'time_low_11_ff': 1278376200.0, 'close_rolling_21d_max': 0.24899188405209483, 'close_high_21_ff': 0.24899188405209483, 'close_high_21_bf': 0.24899188405209483, 'time_high_21_ff': 1280235600.0, 'close_rolling_21d_min': 0.16962714250622124, 'close_low_21_ff': 0.16962714250622124, 'close_low_21_bf': 0.16962714250622124, 'time_low_21_ff': 1278376200.0, 'close_rolling_51d_max': 0.25692910380175515, 'close_high_51_ff': 0.25692910380175515, 'close_high_51_bf': 0.25692910380175515, 'time_high_51_ff': 1272265200.0, 'close_rolling_51d_min': 0.16962714250622124, 'close_low_51_ff': 0.16962714250622124, 'close_low_51_bf': 0.16962714250622124, 'time_low_51_ff': 1278376200.0, 'close_rolling_201d_max': 0.31111140852979613, 'close_high_201_ff': 0.31111140852979613, 'close_high_201_bf': 0.31111140852979613, 'time_high_201_ff': 1272265200.0, 'close_rolling_201d_min': 0.16962714250622124, 'close_low_201_ff': 0.16962714250622124, 'close_low_201_bf': 0.16962714250622124, 'time_low_201_ff': 1278376200.0, 'rsi': 40.72939037110705, 'macd': -2.0581263952477684, 'macd_signal': -2.009451149779616, 'bb_high': 966.673950325657, 'bb_low': 947.0260496743431, 'sma_50': 960.535, 'sma_200': 963.78375, 'daily_returns': 0.00026198585276393516, 'daily_vol': 0.00220817234559232, 'macd_8_24': -1.423010844601806, 'macd_16_48': -1.0358884337028256, 'macd_32_96': 0.12112227452390391, 'week_of_year': 30, 'month_of_year': 7, 'weekly_close_time': 1280523600, 'monthly_close_time': 1280523600, 'option_expiration_time': 1282338000, 'new_york_open_time': 1280410200, 'new_york_last_open_time': 1280410200.0, 'new_york_open_cum_dv': 613424553295.0, 'new_york_open_cum_volume': 691510058.0, 'vwap_since_new_york_open': -0.0014545967872452437, 'vwap_around_new_york_open': 964.3921794675732, 'ret_from_vwap_around_new_york_open': -0.0067780628619651395, 'new_york_close_time': 1280433600, 'london_open_time': 1280386800, 'london_last_open_time': 1280386800.0, 'london_open_cum_dv': 613070157512.25, 'london_open_cum_volume': 691143683.0, 'vwap_since_london_open': -0.0026770785541474496, 'vwap_around_london_open': 963.5447276863375, 'ret_from_vwap_around_london_open': -0.006199189865877308, 'london_close_time': 1280417400, 'time_to_new_york_open': -34200, 'time_to_new_york_last_open': -34200.0, 'time_to_new_york_close': -10800, 'time_to_london_open': -57600, 'time_to_london_last_open': -57600.0, 'time_to_london_close': -27000, 'time_to_weekly_close': 79200, 'time_to_monthly_close': 79200, 'time_to_option_expiration': 1893600, 'time_to_high_5_ff': -208800.0, 'time_to_low_5_ff': -817200.0, 'time_to_high_11_ff': -208800.0, 'time_to_low_11_ff': -2068200.0, 'time_to_high_21_ff': -208800.0, 'time_to_low_21_ff': -2068200.0, 'time_to_high_51_ff': -8179200.0, 'time_to_low_51_ff': -2068200.0, 'time_to_high_201_ff': -8179200.0, 'time_to_low_201_ff': -2068200.0, 'relative_time_idx': 0.0
        }
        logging.error(f"trader.current_data_row:{trader.current_data_row.to_dict()}")
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
