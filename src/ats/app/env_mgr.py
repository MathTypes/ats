from collections import defaultdict
import datetime
import logging

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from pytorch_forecasting.utils import create_mask, detach, to_list
import torch

from ats.calendar import market_time
from ats.market_data.data_module import (
    TransformerDataModule,
    LSTMDataModule,
    TimeSeriesDataModule,
)
from ats.model.models import AttentionEmbeddingLSTM
from ats.model import model_utils
from ats.prediction import prediction_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.optimizer import position_utils
from ats.util.profile import profile
from ats.calendar import market_time


class EnvMgr(object):
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.init_env()

    def init_env(self):
        self.train_start_date = datetime.datetime.strptime(
            self.config.job.train_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.eval_start_date = datetime.datetime.strptime(
            self.config.job.eval_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.eval_end_date = datetime.datetime.strptime(
            self.config.job.eval_end_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.test_start_date = datetime.datetime.strptime(
            self.config.job.test_start_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.test_end_date = datetime.datetime.strptime(
            self.config.job.test_end_date, "%Y-%m-%d"
        ).replace(tzinfo=datetime.timezone.utc)
        self.max_lags = self.config.job.max_lags
        start_date = None
        end_date = None
        if self.config.job.mode == "train":
            self.train_start_date = datetime.datetime.strptime(
                self.config.job.train_start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
        elif self.config.job.mode == "test":
            self.data_start_date = self.test_start_date - datetime.timedelta(
                days=self.max_lags
            )
            self.market_cal = mcal.get_calendar(self.config.job.market)
        logging.info(f"train_start_date:{self.train_start_date}, test_start_date:{self.test_start_date}, test_end_date:{self.test_end_date}")
        start_date = self.train_start_date
        if not start_date:
            start_date = self.data_start_date
        self.start_date = start_date
        self.end_date = self.test_end_date
        logging.info(
            f"start_date:{start_date}, test_start_date:{self.test_start_date}, test_end_date:{self.test_end_date}"
        )
