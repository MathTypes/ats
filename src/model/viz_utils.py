import datetime
from io import BytesIO
import logging

import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, PatchTstTransformer, PatchTstTftTransformer, PatchTftSupervised
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, MAPCSE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.utils import create_mask, detach, to_list
import plotly.graph_objects as go
import PIL
from plotly.subplots import make_subplots
from timeseries_transformer import TimeSeriesTFT
import torch
import wandb

def create_example_viz_table(model, data_loader, eval_data, metrics, top_k):
    wandb_logger = WandbLogger(project='ATS', log_model=True)
    trainer_kwargs = {'logger':wandb_logger}
    raw_predictions = model.predict(
        data_loader,
        mode="raw",
        return_x=True,
        return_index=True,
        return_y=True,
        trainer_kwargs=trainer_kwargs,
    )
    prediction_kwargs = {"reduction": None}
    prediction_kwargs = {}
    y_hats = to_list(
        model.to("cuda:0").to_prediction(
            [None, raw_predictions.output], **prediction_kwargs
        )
    )[0]
    mean_losses = metrics(y_hats, raw_predictions.y).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    matched_eval_data = eval_data
    x = raw_predictions.x
    y_close = raw_predictions.y[0]
    y_close_cum_sum = torch.cumsum(y_close, dim=-1)
    y_hats_cum_sum = torch.cumsum(y_hats, dim=-1)
    column_names = [
        "ticker",
        "time",
        "time_idx",
        "day_of_week",
        "hour_of_day",
        "year",
        "month",
        "day_of_month",
        "price_img",
        "act_close_pct_max",
        "act_close_pct_min",
        "close_back_cumsum",
        "time_str",
        "pred_time_idx",
        "pred_close_pct_max",
        "pred_close_pct_min",
        "img",
        "error_max",
        "error_min",
        "rmse",
        "mae",
    ]
    data_table = wandb.Table(columns=column_names, allow_mixed_types=True)
    day_of_week_map = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]
    for idx in range(top_k):
        idx = indices[idx]
        context_length = len(x["encoder_target"][idx])
        prediction_length = len(x["decoder_time_idx"][idx])
        decoder_time_idx = int(x["decoder_time_idx"][idx][0].cpu().detach().numpy())
        y_close_cum_sum_row = y_close_cum_sum[idx]
        train_data_row = matched_eval_data[
            matched_eval_data.time_idx == decoder_time_idx
        ].iloc[0]
        dm = train_data_row["time"]
        dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
        train_data_rows = matched_eval_data[
            (matched_eval_data.time_idx >= decoder_time_idx - context_length)
            & (matched_eval_data.time_idx < decoder_time_idx + prediction_length)
        ]
        x_time = train_data_rows["time"]
        # logging.info(f"xtime:{x_time}")
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
            ],
        )
        fig.update_layout(
            autosize=False,
            width=1500,
            height=800,
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[17, 4], pattern="hour"),  # hide hours outside of 4am-5pm
            ],
        )
        prediction_date_time = (
            train_data_row["ticker"]
            + " "
            + dm_str
            + " "
            + day_of_week_map[train_data_row["day_of_week"]]
        )
        fig.update_layout(title=prediction_date_time, font=dict(size=20))
        model.plot_prediction(
            raw_predictions.x,
            (None, raw_predictions.output),
            idx=idx,
            # add_loss_to_title=SMAPE(quantiles=self.model.loss.quantiles),
            ax=fig,
            row=1,
            col=1,
            draw_mode="pred_cum",
            x_time=x_time,
        )

        img_bytes = fig.to_image(format="png")  # kaleido library
        im = PIL.Image.open(BytesIO(img_bytes))
        pred_img = wandb.Image(im)

        y_hat = y_hats[idx]
        y_hat_cum = y_hats_cum_sum[idx]
        img = wandb.Image(im)
        fig = go.Figure(
            data=go.Ohlc(
                x=train_data_rows["time"],
                open=train_data_rows["open"],
                high=train_data_rows["high"],
                low=train_data_rows["low"],
                close=train_data_rows["close"],
            )
        )
        # add a bar at prediction time
        fig.update(layout_xaxis_rangeslider_visible=False)
        prediction_date_time = (
            train_data_row["ticker"]
            + " "
            + dm_str
            + " "
            + day_of_week_map[train_data_row["day_of_week"]]
        )
        fig.update_layout(title=prediction_date_time, font=dict(size=20))
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[17, 4], pattern="hour"),  # hide hours outside of 4am-5pm
            ],
        )
        img_bytes = fig.to_image(format="png")  # kaleido library
        im = PIL.Image.open(BytesIO(img_bytes))
        img = wandb.Image(im)
        base = 0
        y_max = torch.max(y_close_cum_sum_row) - base
        y_min = torch.min(y_close_cum_sum_row) - base
        pred_max = torch.max(y_hat_cum)
        pred_min = torch.min(y_hat_cum)
        data_table.add_data(
            train_data_row["ticker"],  # 0 ticker
            dm,  # 1 time
            train_data_row["time_idx"],  # 2 time_idx
            train_data_row["day_of_week"],  # 3 day of week
            train_data_row["hour_of_day"],  # 4 hour of day
            train_data_row["year"],  # 5 year
            train_data_row["month"],  # 6 month
            train_data_row["day_of_month"],  # 7 day_of_month
            wandb.Image(im),  # 8 image
            # np.argmax(label, axis=-1)
            y_max,  # 9 max
            y_min,  # 10 min
            base,  # 11 close_back_cusum
            dm_str,  # 12
            decoder_time_idx,  # 13,
            torch.max(y_hat_cum),  # 14
            torch.min(y_hat_cum),  # 15
            pred_img,  # 16
            pred_max - y_max,  # error_max
            pred_min - y_min,  # error_min
            0,  # 17
            0,  # 18
        )
    return data_table
