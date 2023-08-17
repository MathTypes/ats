from collections import defaultdict
import datetime
import logging

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
    # cum_returns,
)
import numpy as np
import pandas as pd
import pytz
from pytorch_forecasting.utils import detach
import torch

from ats.calendar import market_time
from ats.model import viz_utils
from ats.prediction import prediction_utils
from ats.optimizer import position_utils
from ats.util import profile_util

class Trader(object):
    def __init__(
        self,
        md_mgr,
        model,
        wandb_logger,
        target_size,
        future_data,
        train_data,
        train_dataset,
        config,
        market_cal,
    ):
        super().__init__()
        self.market_data_mgr = md_mgr
        self.last_time_idx = train_data.iloc[-1]["time_idx"]
        #logging.info(f"train_data:{train_data.iloc[-3:]}")
        self.last_data_time = None
        self.last_px_map = {}
        last_data = train_data.iloc[-1]
        #logging.info(f"last_data:{last_data}")
        self.last_px_map[last_data.ticker] = last_data.close
        self.last_position_map = defaultdict(lambda: 0, {})
        self.first_update = True
        initial_positions = torch.tensor([0])
        self.config = config
        #logging.info(f"sigma:{self.config.trading.sigma}")
        self.market_cal = market_cal
        self.model = model
        self.optimizer = position_utils.Optimizer(
            name="opt",
            max_loss=self.config.trading.max_loss,
            gamma=self.config.trading.gamma,
            sigma=self.config.trading.sigma,
            initial_positions=initial_positions,
        )
        self.wandb_logger = wandb_logger
        self.train_dataset = train_dataset
        self.train_data = train_data
        self.target_size = target_size
        self.future_data = future_data
        self.pnl_df = pd.DataFrame(
            columns=["pos", "px", "y_hat_cum_max", "y_hat_cum_min", "pnl", "time", "y_close_cum_max", "y_close_cum_min",
                     "last_px", "ticker"]
        )
        self.cnt = 0

    def compute_stats(self, srs: pd.DataFrame, metric_suffix=""):
        return {
            f"annual_return{metric_suffix}": annual_return(srs),
            f"annual_volatility{metric_suffix}": annual_volatility(srs),
            f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
            f"downside_risk{metric_suffix}": downside_risk(srs),
            f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
            f"max_drawdown{metric_suffix}": -max_drawdown(srs),
            f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
            f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
            f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
            / np.mean(np.abs(srs[srs < 0.0])),
        }

    @profile_util.profile
    def on_interval(self, utc_time):
        if self.last_data_time is not None and utc_time<self.last_data_time + datetime.timedelta(minutes=self.config.job.time_interval_minutes):
            return None
        
        max_prediction_length = self.config.model.prediction_length
        # prediction is at current time, so we need max_prediction_length + 1.
        trading_times = market_time.get_next_trading_times(
            self.market_cal, self.config.job.time_interval_minutes,
            utc_time, max_prediction_length + 1
        )
        predict_nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
        # Do not trade other than between 10 and 16 NYC time.
        if predict_nyc_time.hour < 10 or predict_nyc_time.hour>=16:
            return None
        #logging.error(f"trading_times:{trading_times}")
        new_data = self.future_data[
            (self.future_data.timestamp >= trading_times[0])
            & (self.future_data.timestamp <= trading_times[-1])
            & (self.future_data.ticker == "ES")
        ]
        train_raw_data = self.train_dataset.raw_data[
            (self.train_dataset.raw_data.timestamp < trading_times[0])
            & (self.train_dataset.raw_data.ticker == "ES")
        ]
        #logging.error(f"new_data_from_future:{new_data.iloc[:10][['time','timestamp','close_back']]}")
        missing_times = len(trading_times) - len(new_data)
        #logging.error(f"missing_times:{missing_times}, trading_times:{len(trading_times)}, new_data:{len(new_data)}")
        if missing_times>0:
            starting_trading_times = len(new_data)
            new_data_df = pd.DataFrame(columns=["open","close","high","low","volume","dv","ticker","new_idx"])
            for idx in range(missing_times):
                timestamp = int(trading_times[starting_trading_times+idx])
                time = datetime.datetime.utcfromtimestamp(timestamp).astimezone(pytz.timezone("America/New_York"))
                logging.error(f"adding timestamp:{timestamp}, idx:{idx}, starting_trading_times:{starting_trading_times}")
                new_row = {"time":time,
                           "timestamp":timestamp,
                           "open":0.1, "close":0.1, "high":0.1, "low":0.1, "volume":1, "dv":1,
                           "series_idx":str(time),
                           "ticker":"ES","new_idx":"ES_" + str(timestamp)}
                new_data_df = pd.concat([new_data_df, pd.DataFrame(new_row, index=["new_idx"])])
            new_data_df = new_data_df.set_index("new_idx")
            new_data = pd.concat([new_data, new_data_df])
        #logging.error(f"new_data:{new_data}")
        #bad_new_data = new_data[new_data.isna().any(axis=1)]
        #if not bad_new_data.empty:
        #    logging.error(f"bad_new_data:{bad_new_data}")
        #    exit(0)
        if new_data.empty:
            if (
                self.last_data_time is None
                or predict_nyc_time
                < self.last_data_time
                + datetime.timedelta(minutes=self.config.job.max_stale_minutes)
            ):
                return None
            else:
                logging.info(
                    f"data is too stale, now:{predict_nyc_time}, last_data_time:{self.last_data_time}"
                )
                return None
        self.last_data_time = predict_nyc_time
        self.last_time_idx = train_raw_data.iloc[-1]["time_idx"]
        self.last_time_idx += 1
        new_data["time_idx"] = range(
            self.last_time_idx, self.last_time_idx + len(new_data)
        )
        logging.info(f"running step {predict_nyc_time}, new_data:{new_data.iloc[:3][['time_idx','close','time']]}")
        self.train_dataset.add_new_data(
            new_data,
            self.config.job.time_interval_minutes,
            self.market_cal,
            self.market_data_mgr,
        )
        predict_time_idx_end = self.last_time_idx + max_prediction_length
        # logging.error(f"new_train_dataset:{train_dataset.raw_data[-3:]}")
        logging.error(
            f"last_time_idex={self.last_time_idx}, predict_time_idx_end:{predict_time_idx_end}"
        )
        filtered_dataset = self.train_dataset.filter(
            lambda x: (x.time_idx_last == predict_time_idx_end)
        )
        x, y = next(iter(filtered_dataset.to_dataloader(train=False, batch_size=1)))
        #logging.info(f"x:{x}")
        #logging.info(f"y:{y}")
        # new_prediction_data is the last encoder_data, we need to add decoder_data based on
        # known features or lagged unknown features
        # logging.info(f"new_prediction_data:{new_prediction_data}")
        y_hats, y_quantiles, out, x = prediction_utils.predict(
            self.model, filtered_dataset, self.wandb_logger, batch_size=1
        )
        #logging.info(f"y_hats:{y_hats}")
        #logging.info(f"y_quantiles:{y_quantiles}")
        #logging.info(f"out:{out}")
        if isinstance(y_hats, list):
            y_hats = y_hats[0]
        returns_fcst = y_hats.cpu().numpy()
        min_y_quantiles = y_quantiles[:, :, 0]
        max_y_quantiles = y_quantiles[:, :, -1]
        logging.info(f"min_y_quantiles:{min_y_quantiles}, max_y_quantiles:{max_y_quantiles}")
        cum_min_y_quantiles = torch.cumsum(min_y_quantiles, 1)
        cum_max_y_quantiles = torch.cumsum(max_y_quantiles, 1)
        logging.info(f"cum_min_y_quantiles:{cum_min_y_quantiles}, cum_max_y_quantiles:{cum_max_y_quantiles}")
        min_fcst = torch.min(cum_min_y_quantiles)
        max_fcst = torch.max(cum_max_y_quantiles)
        logging.info(f"min_fcst:{min_fcst}, max_fcst:{max_fcst}")
        min_neg_fcst = (
            torch.minimum(min_fcst, torch.tensor(0)).unsqueeze(0).detach().cpu().numpy()
        )
        max_pos_fcst = (
            torch.maximum(max_fcst, torch.tensor(0)).unsqueeze(0).detach().cpu().numpy()
        )
        logging.info(f"returns_fcst:{returns_fcst}, min_neg_fcst:{min_neg_fcst}, max_pos_fcst:{max_pos_fcst}")
        new_positions, ret, val = self.optimizer.optimize(
            returns_fcst, min_neg_fcst, max_pos_fcst
        )
        #logging.info(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
        y_hats_cum = torch.cumsum(y_hats, dim=-1)
        logging.info(f"y:{y}")
        # y is open/high/low/close
        y_close = y[0][0]
        y_close_cum_sum = torch.cumsum(y_close, dim=-1)
        # logging.info(f"x:{x}")
        # self.last_time_idx is current interval (as of close is known). indices would be self.last_time_idx + 1
        indices = self.train_dataset.x_to_index(x)
        matched_data = self.train_dataset.raw_data
        #logging.error(f"indices:{indices}")
        rmse = [0]
        mae = [0]
        interp_output = self.model.interpret_output(
            detach(out),
            reduction="none",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        # logging.info(f"interp_output:{interp_output}")
        log_viz = self.cnt % self.config.trading.log_viz_every_n == 0
        # We always predict at t+1. So row would be at t+1.
        row = viz_utils.create_viz_row(
            0,
            y_hats,
            y_hats_cum,
            y_close,
            y_close_cum_sum,
            indices,
            matched_data,
            x,
            self.config,
            self.model,
            out,
            self.target_size,
            interp_output,
            rmse,
            mae,
            filter_small=False,
            show_viz=log_viz,
        )
        # new_data is close price of current row. 
        new_data_row = new_data.iloc[0]
        logging.info(f"return from viz_row:{row}, new_data:{new_data_row}")
        ticker = new_data_row.ticker
        px = new_data_row.close
        last_position = self.last_position_map[ticker]
        last_px = self.last_px_map[ticker]
        pnl_delta = last_position * (px - last_px)
        new_position = new_positions[0]
        row["last_position"] = last_position
        row["new_position"] = new_position
        row["delta_position"] = new_position - last_position
        row["px"] = px
        row["pnl_delta"] = pnl_delta
        new_pnl_row = {
            "pos": new_position,
            "px": px,
            "y_hat_cum_max":row["y_hat_cum_max"],
            "y_hat_cum_min":row["y_hat_cum_min"],
            "pnl": pnl_delta,
            "time": new_data_row.time,
            "y_close_cum_max":row["y_close_cum_max"],
            "y_close_cum_min":row["y_close_cum_min"],
            "last_px": last_px,
            "ticker": ticker,
        }
        logging.info(f"new_pnl_row:{new_pnl_row}")
        self.current_data_row = new_data_row
        self.pnl_df.loc[len(self.pnl_df)] = new_pnl_row
        #self.pnl_df = self.pnl_df.append(df2, ignore_index=True)
        self.last_position_map[ticker] = new_position
        self.last_px_map[ticker] = px
        logging.info(f"return row:{row}")
        logging.info(f"last_position_map:{self.last_position_map}")
        logging.info(f"last_px_map:{self.last_px_map}")
        if not self.pnl_df.empty:
            stats = self.compute_stats(self.pnl_df.pnl/500)
            logging.info(f"stats:{stats}")
            #logging.info(f"pnl_df:{self.pnl_df}")
        del filtered_dataset
        del y_hats, y_quantiles, out, x
        return row
