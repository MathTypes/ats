import datetime
import logging

import pandas as pd

from ats.calendar import date_utils
from ats.calendar import market_time

class MacroDataBuilder:
    def __init__(self, env_mgr):
        self.env_mgr = env_mgr
        self.config = env_mgr.config
        self.add_macro_event = self.config.model.features.add_macro_event
        if self.add_macro_event:
            self.load_forex_factory_events()

    def load_forex_factory_events(self):
        df_vec = []
        for begin, end in date_utils.monthlist(self.env_mgr.data_start_date, self.env_mgr.data_end_date):
            date_str = begin.strftime("%Y.%m.%d")
            # 2023.08.01 00:30:00;AUD;3;Cash Rate;4.10%;4.35%;4.10%;;;
            observations = pd.read_csv(
                f"/home/ubuntu/ats/data/event/forex_factory/{date_str}.csv",
                index_col=False, sep=";", header=None,
                names=["date","currency","importance","id","actual","forecast","previous"]
            )
            observations.date = observations.date.apply(
                lambda x: datetime.datetime.strptime(x, "%Y.%m.%d %H:%M:%S")
            )
            observations.date = observations.date.apply(
                market_time.utc_to_nyse_time,
                interval_minutes=self.config.job.time_interval_minutes,
            )
            observations["event_time"] = observations.date.apply(lambda x:x.timestamp())
            observations["importance"] = observations.importance.apply(lambda x:int(x))
            df_vec.append(observations)
        self.observations = pd.concat(df_vec)
        self.observations = self.observations.set_index(["date"])
        self.observations = self.observations.sort_index()
        #logging.error(f"self.observations:{self.observations.iloc[-3:]}")
        self.observations = self.observations[(self.observations.currency=="USD") & (self.observations.importance>2)]
        #logging.error(f"after self.observations:{self.observations.iloc[-3:]}")

    def load_events(self):
        df_vec = []
        for begin, end in date_utils.monthlist(start_date, end_date):
            year_month_str = begin.strftime("%Y-%m")
            date_str = begin.strftime("%Y-%m-%d")
            observations = pd.read_csv(
                f"/home/ubuntu/ats/data/event/macro/series_observation/{year_month_str}/{date_str}.csv",
                sep="~",
            )
            observations.date = observations.date.apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
            )
            observations["event_time"] = observations.date.apply(
                lambda x: datetime.datetime.combine(
                    x, datetime.datetime.min.time()
                ).timestamp()
            )
        observations = pd.concat(df_vec)
        self.observations = observations[["date", "value", "series_id", "event_time"]]

    def get_last_events(self, dt):
        df = self.observations[
            (self.observations.event_time <= dt)
            & (self.observations.event_time > dt - 2 * 86400000)
        ]
        return df

    def get_next_events(self, dt):
        df = self.observations[
            (self.observations.event_time < dt+2*86400000)
            & (self.observations.event_time >= dt)
        ]
        return df

    def has_event(self, dt):
        df = self.observations[
            (self.observations.event_time < dt.timestamp())
            & (self.observations.event_time > dt.timestamp() - 86400000)
        ]
        return not df.empty
