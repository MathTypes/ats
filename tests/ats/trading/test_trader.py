import logging

import pandas as pd


from hydra import initialize, compose

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr
from ats.model import model_utils
from ats.trading.trader import Trader


# THe config path is relative to the file calling initialize (this file)
def test_on_interval():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        market_cal = md_mgr.market_cal
        wandb_logger = None
        data_module = md_mgr.data_module
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
        logging.error(f"train_data start:{train_data[:5]['time']}")
        logging.error(f"train_data end:{train_data[-5:]['time']}")
        logging.error(f"future_data start:{future_data[:2]}")
        logging.error(f"future_data end:{future_data[-2:]}")
