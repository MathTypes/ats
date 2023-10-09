import datetime
from functools import partial
import logging
import math

from hydra import initialize, compose
import numpy as np
import pandas as pd
import ray

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
    # cum_returns,                                                                                                                                                )
)

from ats.app.env_mgr import EnvMgr
from ats.event.macro_indicator import MacroDataBuilder
from ats.market_data import data_util
from ats.market_data import market_data_mgr
from ats.util import logging_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:.5f}".format

