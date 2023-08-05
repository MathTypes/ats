import logging

import pandas as pd


from hydra import initialize, compose

from ats.app.env_mgr import EnvMgr
from ats.market_data import market_data_mgr


# THe config path is relative to the file calling initialize (this file)
def test_get_snapshot():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test",
            overrides=[
                "dataset.snapshot=''",
                "dataset.write_snapshot=False",
            ],
            return_hydra_config=True
        )
        env_mgr = EnvMgr(cfg)
        #md_mgr = market_data_mgr.MarketDataMgr(env_mgr, simulation_mode=True)
        md_mgr = market_data_mgr.MarketDataMgr(env_mgr)
        # wandb_logger = WandbLogger(project="ATS", log_model=True)
        raw_data = md_mgr.full_data()
        first_data_row = raw_data.iloc[0]
        # 2008-02-07 08:30:00 - we need 201 prior intervals. Each day
        # has about 40 intervals. It is strange why we go to 2008-02-07 instead
        # of 2008-01-10.
        # Fri Mar 28 2008 04:30:00
        assert first_data_row["timestamp"] == 1238365800
        data_module = md_mgr.data_module()
        first_train_data_row = data_module.train_data.iloc[0]
        #  Tue Jun 30 2009 22:00:00
        assert first_train_data_row["timestamp"] == 1243807200
        last_train_data_row = data_module.train_data.iloc[-1]
        # Tue Jun 30 2009 22:00:00 UTC
        assert last_train_data_row["timestamp"] == 1246399200
        first_eval_data_row = data_module.eval_data.iloc[0]
        # 2009-06-30 18:00:00
        assert first_eval_data_row["timestamp"] == 1246399200
        last_eval_data_row = data_module.eval_data.iloc[-1]
        # 2009-07-30 21:30:00.
        assert last_eval_data_row["timestamp"] == 1249333200
        first_test_data_row = data_module.test_data.iloc[0]
        assert first_test_data_row["timestamp"] == 1246399200
        last_test_data_row = data_module.test_data.iloc[-1]
        assert last_test_data_row["timestamp"] == 1249333200
