from collections import defaultdict
import datetime
import logging

import pandas as pd
import pytz
from pytorch_forecasting.utils import create_mask, detach, to_list
import torch

from ats.calendar import market_time
from ats.market_data.data_module import (
    TransformerDataModule,
    LSTMDataModule,
    TimeSeriesDataModule,
)
from ats.event.macro_indicator import MacroDataBuilder
from ats.model.models import AttentionEmbeddingLSTM
from ats.model import model_utils
from ats.prediction import prediction_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.optimizer import position_utils
from ats.util.profile import profile
from ats.calendar import market_time

class MarketDataMgr(object):
    def __init__(
        self,
        config,
        market_cal,
    ):
        super().__init__()
        self.config = config
        self.market_cal = market_cal
        self.macro_data_builder = MacroDataBuilder(config)

    
