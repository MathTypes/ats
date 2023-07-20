import datetime
import logging

import pytz
from pytorch_forecasting.utils import create_mask, detach, to_list
import torch

from ats.calendar import market_time
from ats.model.data_module import TransformerDataModule, LSTMDataModule, TimeSeriesDataModule
from ats.model.models import AttentionEmbeddingLSTM
from ats.model import model_utils
from ats.prediction import prediction_utils
from ats.model.utils import Pipeline
from ats.model import viz_utils
from ats.optimizer import position_utils
from ats.util.profile import profile
from ats.util import trace_utils
from ats.calendar import market_time

def on_interval(
        model, optimizer, wandb_logger,
        future_data, train_dataset, last_time_idx, config,
        last_position_map, last_px_map, pnl_df, first_update, market_cal, utc_time):
    max_prediction_length = config.model.prediction_length
    # prediction is at current time, so we need max_prediction_length + 1.
    trading_times = market_time.get_next_trading_times(
        market_cal, "30M", utc_time, max_prediction_length + 1
    )
    predict_nyc_time = utc_time.astimezone(pytz.timezone("America/New_York"))
    logging.info(f"utc_time:{utc_time}, trading_times:{trading_times}")
    trace_utils.take_snapshot()
    if first_update:
        logging.info(f"future_data:{future_data.iloc[:3]}")
        new_data = future_data[
            (future_data.timestamp >= trading_times[0])
            & (future_data.timestamp <= trading_times[-1])
            & (future_data.ticker == "ES")
        ]
        first_update = False
    else:
        new_data = future_data[
            (future_data.timestamp == trading_times[-1]) & (future_data.ticker == "ES")
        ]
    if new_data.empty:
        if (
            last_data_time is None
            or predict_nyc_time
            < last_data_time
            + datetime.timedelta(minutes=config.dataset.max_stale_minutes)
        ):
            return None
        else:
            logging.info(
                f"data is too stale, now:{predict_nyc_time}, last_data_time:{last_data_time}"
            )
            return None
    last_data_time = predict_nyc_time
    last_time_idx += 1
    new_data["time_idx"] = range(last_time_idx, last_time_idx + len(new_data))
    logging.info(f"running step {predict_nyc_time}")
    train_dataset.add_new_data(new_data, config.job.time_interval_minutes)
    predict_time_idx = new_data.time_idx.max()
    # logging.info(f"new_train_dataset:{train_dataset.raw_data[-3:]}")
    logging.info(f"last_time_idex={last_time_idx}, predict_time_idx:{predict_time_idx}")
    filtered_dataset = train_dataset.filter(
        lambda x: (x.time_idx_last == predict_time_idx)
    )
    x, y = next(iter(filtered_dataset.to_dataloader(train=False, batch_size=1)))
    # logging.info(f"x:{x}, y:{y}")
    # new_prediction_data is the last encoder_data, we need to add decoder_data based on
    # known features or lagged unknown features
    # logging.info(f"new_prediction_data:{new_prediction_data}")
    y_hats, y_quantiles, out = prediction_utils.predict(
        model, filtered_dataset, wandb_logger
    )
    #logging.info(f"y_hats:{y_hats}")
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
    new_positions, ret, val = optimizer.optimize(
        returns_fcst, min_neg_fcst, max_pos_fcst
    )
    logging.info(f"new_positions:{new_positions}, ret:{ret}, val:{val}")
    y_hats_cum = torch.cumsum(y_hats, dim=-1)
    y_close = y[0]
    y_close_cum_sum = torch.cumsum(y_close, dim=-1)
    # logging.info(f"x:{x}")
    indices = train_dataset.x_to_index(x)
    matched_data = train_dataset.raw_data
    # logging.info(f"indices:{indices}")
    rmse = [0]
    mae = [0]
    interp_output = model.interpret_output(
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
        data_table,
        config,
        model,
        out,
        target_size,
        interp_output,
        rmse,
        mae,
    )
    new_data_row = new_data.iloc[0]
    ticker = new_data_row.ticker
    px = new_data_row.close
    last_position = last_position_map[ticker]
    last_px = last_px_map[ticker]
    pnl_delta = last_position * (px - last_px)
    new_position = new_positions[0]
    df2 = {
        "ticker": ticker,
        "timestamp": new_data_row.timestamp,
        "px": px,
        "last_px": last_px,
        "pos": new_position,
        "pnl": pnl_delta,
    }
    logging.info(f"new_df:{df2}")
    pnl_df = pnl_df.append(df2, ignore_index=True)
    last_position_map[ticker] = new_position
    last_px_map[ticker] = px
    return row
