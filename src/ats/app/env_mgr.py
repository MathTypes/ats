import datetime
import logging
from typing import List

import pandas_market_calendars as mcal

from ats.calendar import market_time
from ats.model import model_utils


class EnvMgr(object):
    def __init__(
        self,
        config,
        run_id = None
    ):
        self.config = config
        self.run_id = run_id
        self.init_env()

    def init_env(self):
        self.market_cal = mcal.get_calendar(self.config.job.market)
        has_train_stage = self.config.job.mode in ["train", "test"]
        has_eval_stage = self.config.job.mode in ["train", "test", "eval", "build_search"]
        has_test_stage = self.config.job.mode in ["test"]

        self.max_lags = self.config.job.max_lag
        if has_train_stage:
            self.train_start_date = datetime.datetime.strptime(
                self.config.job.train_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.train_start_timestamp = market_time.get_open_time(
                self.market_cal, self.train_start_date
            )
        if has_eval_stage:
            self.eval_start_date = datetime.datetime.strptime(
                self.config.job.eval_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_start_timestamp = market_time.get_open_time(
                self.market_cal, self.eval_start_date
            )
            self.eval_end_date = datetime.datetime.strptime(
                self.config.job.eval_end_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_end_timestamp = market_time.get_close_time(
                self.market_cal, self.eval_end_date
            )

        if has_test_stage:
            self.test_start_date = datetime.datetime.strptime(
                self.config.job.test_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_end_timestamp = market_time.get_close_time(
                self.market_cal, self.eval_end_date
	    )
            self.test_end_date = datetime.datetime.strptime(
                self.config.job.test_end_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.test_start_timestamp = market_time.get_open_time(
                self.market_cal, self.test_start_date
            )
            self.test_end_timestamp = market_time.get_close_time(
                self.market_cal, self.test_end_date
            )
            self.data_start_date = self.test_start_date - datetime.timedelta(
                days=self.max_lags
            )
        data_start_date = None
        data_end_date = None
        if self.config.job.mode == "train":
            data_start_date = self.train_start_date
            data_end_date = self.test_end_date
        elif self.config.job.mode == "test":
            data_start_date = self.train_start_date
            data_end_date = self.test_end_date
        elif self.config.job.mode == "eval":
            data_start_date = self.eval_start_date
            data_end_date = self.eval_end_date
            # Still need to fail train/test time since data_module
            # build train/eval/test dataset
            self.train_start_timestamp = self.eval_start_timestamp
            self.test_start_timestamp = self.eval_start_timestamp
            self.test_end_timestamp = self.eval_end_timestamp
        elif self.config.job.mode == "build_search":
            data_start_date = self.eval_start_date
            data_end_date = self.eval_end_date
            # Still need to fail train/test time since data_module
            # build train/eval/test dataset
            self.train_start_timestamp = self.eval_start_timestamp
            self.test_start_timestamp = self.eval_start_timestamp
            self.test_end_timestamp = self.eval_end_timestamp
            
        self.data_start_date = data_start_date - datetime.timedelta(days=self.config.job.max_lag)
        self.data_end_date = data_end_date + datetime.timedelta(days=self.config.job.max_lead)
        self.dataset_base_dir = self.config.dataset.base_dir
        self.model_tickers = self.config.dataset.model_tickers
        self.heads, self.targets = model_utils.get_heads_and_targets(self.config)
        self.time_interval = self.config.dataset.time_interval
        self.target_size = len(self.targets) if isinstance(self.targets, List) else 1
        self.context_length = self.config.model.context_length
        self.prediction_length = self.config.model.prediction_length
        logging.info(
            f"data_start_date:{self.data_start_date}, data_end_date:{self.data_end_date}"
        )
