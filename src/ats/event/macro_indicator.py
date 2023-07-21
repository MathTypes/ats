import datetime
import logging

import requests
import pandas as pd

from ats.calendar import date_utils
from ats.util import logging_utils

class MacroDataBuilder:
    def __init__(self, config):
        self.config = config
        start_date = datetime.datetime.strptime(config.job.train_start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(config.job.test_end_date, "%Y-%m-%d")
        self.add_macro_event = config.features.add_macro_event
        if add_macro_event:
            self.load_events()
        
    def load_events(self):
        df_vec = []
        for begin, end in date_utils.monthlist(start_date, end_date):
            year_month_str = begin.strftime("%Y-%m")
            date_str = begin.strftime("%Y-%m-%d")
            observations = pd.read_csv(
                f"/home/ubuntu/ats/data/event/macro/series_observation/{year_month_str}/{date_str}.csv",
                sep="~")
            observations.date = observations.date.apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d"))
            logging.info(f"observations:{observations.iloc[-3:]}")
            observations["event_time"] = observations.date.apply(
                lambda x: datetime.datetime.combine(x, datetime.datetime.min.time()).timestamp())
        observations = pd.concat(df_vec)
        self.observations = observations[["date","value","series_id", "event_time"]]
        logging.info(f"observations:{self.observations.iloc[-3:]}")

    def has_event(self, dt):
        df = self.observations[(self.observations.event_time<dt.timestamp()) &
                               (self.observations.event_time>dt.timestamp()-86400000)]
        return not df.empty

