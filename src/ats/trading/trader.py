import datetime
import logging

import pandas as pd
import pytz
from pytorch_forecasting.utils import create_mask, detach, to_list
import torch

from ats.calendar import market_time
from ats.model.data_module import (
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


class Trader(object):
    def __init__(
        self,
        model,
        optimizer,
        wandb_logger,
        target_size,
        future_data,
        train_dataset,
        config,
        last_time_idx,
        last_data_time,
        last_position_map,
        last_px_map,
        market_cal,
    ):
        super().__init__()
        self.config = config
        self.last_data_time = last_data_time
        self.last_time_idx = last_time_idx
        self.last_position_map = last_position_map
        self.last_px_map = last_px_map
        self.market_cal = market_cal
        self.model = model
        self.optimizer = optimizer
        self.wandb_logger = wandb_logger
        self.train_dataset = train_dataset
        self.future_data = future_data
        self.first_update = False
        self.pnl_df = pd.DataFrame(
            columns=["ticker", "timestamp", "px", "last_px", "pos", "pnl"]
        )

    def on_interval(self, utc_time):
        max_prediction_length = self.config.model.prediction_length
        # prediction is at current time, so we need max_prediction_length + 1.
        trading_times = market_time.get_next_trading_times(
            self.market_cal, "30M", utc_time, max_prediction_length + 1
        )
        predict_nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
        logging.info(f"utc_time:{utc_time}, trading_times:{trading_times}")
        if self.first_update:
            # logging.info(f"future_data:{future_data.iloc[:3]}")
            new_data = self.future_data[
                (self.future_data.timestamp >= trading_times[0])
                & (self.future_data.timestamp <= trading_times[-1])
                & (self.future_data.ticker == "ES")
            ]
        else:
            new_data = self.future_data[
                (self.future_data.timestamp == trading_times[-1])
                & (self.future_data.ticker == "ES")
            ]
            self.first_update = False
        if new_data.empty:
            if (
                self.last_data_time is None
                or predict_nyc_time
                < self.last_data_time
                + datetime.timedelta(minutes=self.config.dataset.max_stale_minutes)
            ):
                return None
            else:
                logging.info(
                    f"data is too stale, now:{predict_nyc_time}, last_data_time:{last_data_time}"
                )
                return None
        self.last_data_time = predict_nyc_time
        self.last_time_idx += 1
        new_data["time_idx"] = range(
            self.last_time_idx, self.last_time_idx + len(new_data)
        )
        logging.info(f"running step {predict_nyc_time}")
        self.train_dataset.add_new_data(new_data, self.config.job.time_interval_minutes)
        predict_time_idx = new_data.time_idx.max()
        # logging.info(f"new_train_dataset:{train_dataset.raw_data[-3:]}")
        logging.info(
            f"last_time_idex={self.last_time_idx}, predict_time_idx:{predict_time_idx}"
        )
        filtered_dataset = self.train_dataset.filter(
            lambda x: (x.time_idx_last == predict_time_idx)
        )
        x, y = next(iter(filtered_dataset.to_dataloader(train=False, batch_size=1)))
        # logging.info(f"x:{x}, y:{y}")
        # new_prediction_data is the last encoder_data, we need to add decoder_data based on
        # known features or lagged unknown features
        # logging.info(f"new_prediction_data:{new_prediction_data}")
        y_hats, y_quantiles, out = prediction_utils.predict(
            self.model, filtered_dataset, self.wandb_logger
        )
        # logging.info(f"y_hats:{y_hats}")
        if isinstance(y_hats, list):
            y_hats = y_hats[0]
        returns_fcst = y_hats.numpy()
        min_y_quantiles = y_quantiles[:, :, 0]
        max_y_quantiles = y_quantiles[:, :, -1]
        # logging.info(f"min_y_quantiles:{min_y_quantiles}, max_y_quantiles:{max_y_quantiles}")
        cum_min_y_quantiles = torch.cumsum(min_y_quantiles, 1)
        cum_max_y_quantiles = torch.cumsum(max_y_quantiles, 1)
        # logging.info(f"cum_min_y_quantiles:{cum_min_y_quantiles}, cum_max_y_quantiles:{cum_max_y_quantiles}")
        min_fcst = torch.min(cum_min_y_quantiles)
        max_fcst = torch.max(cum_max_y_quantiles)
        # logging.info(f"min_fcst:{min_fcst}, max_fcst:{max_fcst}")
        min_neg_fcst = (
            torch.minimum(min_fcst, torch.tensor(0)).unsqueeze(0).detach().numpy()
        )
        max_pos_fcst = (
            torch.maximum(max_fcst, torch.tensor(0)).unsqueeze(0).detach().numpy()
        )
        # logging.info(f"returns_fcst:{returns_fcst}, min_neg_fcst:{min_neg_fcst}, max_pos_fcst:{max_pos_fcst}")
        new_positions, ret, val = self.optimizer.optimize(
            returns_fcst, min_neg_fcst, max_pos_fcst
        )
        logging.info(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
        y_hats_cum = torch.cumsum(y_hats, dim=-1)
        y_close = y[0]
        y_close_cum_sum = torch.cumsum(y_close, dim=-1)
        # logging.info(f"x:{x}")
        indices = self.train_dataset.x_to_index(x)
        matched_data = self.train_dataset.raw_data
        # logging.info(f"indices:{indices}")
        rmse = [0]
        mae = [0]
        interp_output = self.model.interpret_output(
            detach(out),
            reduction="none",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        # logging.info(f"interp_output:{interp_output}")
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
        )
        logging.info(f"return from viz_row:{row}")
        new_data_row = new_data.iloc[0]
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
        df2 = {
            "ticker": ticker,
            "timestamp": new_data_row.timestamp,
            "px": px,
            "last_px": last_px,
            "pos": new_position,
            "pnl": pnl_delta,
        }
        logging.info(f"new_df:{df2}")
        self.pnl_df = self.pnl_df.append(df2, ignore_index=True)
        self.last_position_map[ticker] = new_position
        self.last_px_map[ticker] = px
        logging.info(f"return row:{row}")
        return row
