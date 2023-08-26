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
        has_train_stage = self.config.job.mode in ["train", "test", "tune"]
        has_eval_stage = self.config.job.mode in ["train", "tune", "test", "eval", "build_search", "search"]
        has_test_stage = self.config.job.mode in ["test"]

        self.max_lags = self.config.model.max_lag
        if has_train_stage:
            self.train_start_date = datetime.datetime.strptime(
                self.config.job.train_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.train_start_timestamp = market_time.compute_last_open_time(
                self.train_start_date.timestamp(), self.market_cal
            )
        if has_eval_stage:
            self.eval_start_date = datetime.datetime.strptime(
                self.config.job.eval_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_start_timestamp = market_time.compute_last_open_time(
                self.eval_start_date.timestamp(), self.market_cal
            )
            self.eval_end_date = datetime.datetime.strptime(
                self.config.job.eval_end_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_end_timestamp = market_time.compute_next_close_time(
                self.eval_end_date.timestamp(), self.market_cal
            )

        if has_test_stage:
            self.test_start_date = datetime.datetime.strptime(
                self.config.job.test_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.eval_end_timestamp = market_time.compute_next_close_time(
                self.eval_end_date.timestamp(), self.market_cal
	    )
            self.test_end_date = datetime.datetime.strptime(
                self.config.job.test_end_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
            self.test_start_timestamp = market_time.compute_last_open_time(
                self.test_start_date.timestamp(), self.market_cal
            )
            self.test_end_timestamp = market_time.compute_next_close_time(
                self.test_end_date.timestamp(), self.market_cal
            )
            self.data_start_date = self.test_start_date - datetime.timedelta(
                days=self.max_lags
            )
        data_start_date = None
        data_end_date = None
        if self.config.job.mode in ["train", "tune"]:
            data_start_date = self.train_start_date
            data_end_date = self.eval_end_date
            self.test_start_date = self.eval_start_date
            self.test_end_date = self.eval_end_date
            self.test_start_timestamp = self.eval_start_timestamp
            self.test_end_timestamp = self.eval_end_timestamp
        elif self.config.job.mode == "test":
            data_start_date = self.train_start_date
            data_end_date = self.test_end_date
            if data_end_date < self.eval_end_date:
                data_end_date = self.eval_end_date
        elif self.config.job.mode == "eval":
            data_start_date = self.eval_start_date
            data_end_date = self.eval_end_date
            # Still need to fail train/test time since data_module
            # build train/eval/test dataset
            self.train_start_timestamp = self.eval_start_timestamp
            self.test_start_timestamp = self.eval_start_timestamp
            self.test_end_timestamp = self.eval_end_timestamp
        elif self.config.job.mode in ["build_search", "search"]:
            data_start_date = self.eval_start_date
            data_end_date = self.eval_end_date
            # Still need to fail train/test time since data_module
            # build train/eval/test dataset
            # A quick hack to give train data 30 days so that timeseries
            # dataset does not cry.
            self.train_start_timestamp = self.eval_start_timestamp
            self.eval_start_timestamp = self.train_start_timestamp+90*60*60*24
            self.test_start_timestamp = self.eval_start_timestamp
            self.test_end_timestamp = self.eval_end_timestamp
            
        self.data_start_date = data_start_date - datetime.timedelta(days=self.config.model.max_lag)
        self.data_end_date = data_end_date + datetime.timedelta(days=self.config.model.max_lead)
        self.dataset_base_dir = self.config.dataset.base_dir
        self.model_tickers = self.config.dataset.model_tickers
        self.heads, self.all_targets, self.targets = model_utils.get_heads_and_targets(self.config)
        logging.info(f"heads:{self.heads}")
        logging.info(f"targets:{self.targets}")
        logging.info(f"all_targets:{self.all_targets}")
        self.time_interval = self.config.dataset.time_interval
        self.target_size = len(self.targets) if isinstance(self.targets, List) else 1
        self.context_length = self.config.model.context_length
        self.prediction_length = self.config.model.prediction_length
        logging.info(
            f"data_start_date:{self.data_start_date}, data_end_date:{self.data_end_date}"
        )
