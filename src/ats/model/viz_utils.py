import datetime
from io import BytesIO
import logging

from lightning.pytorch.loggers import WandbLogger
from pytorch_forecasting.utils import detach, to_list
import plotly.graph_objects as go
import PIL
from plotly.subplots import make_subplots
import torch
import wandb

from ats.prediction import prediction_data

day_of_week_map = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]


def create_example_viz_table(model, data_loader, eval_data, metrics, top_k):
    wandb_logger = WandbLogger(project="ATS", log_model=True)
    trainer_kwargs = {"logger": wandb_logger}
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
        model.to("cuda:0").to_prediction(raw_predictions.output, **prediction_kwargs)
    )
    logging.info(
        f"y_hats[0].shape:{y_hats[0].shape}, raw_predictions.y:{raw_predictions.y[0].shape}"
    )
    # if isinstance(y_hats, (Tuple)):
    # yhats: [returns, position]
    y_hats = y_hats[0]
    mean_losses = metrics(y_hats, raw_predictions.y).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    logging.info(f"indices:{indices}")
    matched_eval_data = eval_data
    x = raw_predictions.x
    logging.info(f"x:{x['encoder_cont'].shape}")
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
    logging.info(f"x['encoder_target']:{x['encoder_target'][0].shape}")
    interp_output = model.interpret_output(
        detach(raw_predictions.output),
        reduction="none",
        attention_prediction_horizon=0,  # attention only for first prediction horizon
    )
    for k_idx in range(top_k):
        idx = indices[k_idx]
        logging.info(f"k_idx:{k_idx}, idx:{idx}")
        # With multi-target, we have two encoder_targets.
        context_length = len(x["encoder_target"][0][idx])
        prediction_length = len(x["decoder_time_idx"][idx])
        decoder_time_idx = int(x["decoder_time_idx"][idx][0].cpu().detach().numpy())
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
        # y_close = train_data_rows["close_back"][-prediction_length:]
        # logging.info(f"xtime:{x_time}, y_close:{y_close}, y_close_cum_row:{y_close_cum_row}")
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
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
                # dict(bounds=[17, 4], pattern="hour"),  # hide hours outside of 4am-5pm
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
            raw_predictions.output,
            idx=idx,
            ax=fig,
            row=1,
            col=1,
            draw_mode="pred_cum",
            x_time=x_time,
        )
        model.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=idx,
            ax=fig,
            row=2,
            col=1,
            draw_mode="pred_pos",
            x_time=x_time,
        )
        interpretation = {}
        for name in interp_output.keys():
            if interp_output[name].dim() > 1:
                interpretation[name] = interp_output[name][idx]
            else:
                interpretation[name] = interp_output[name]
        model.plot_interpretation(
            interpretation,
            ax=fig,
            cells=[
                {"row": 1, "col": 2},
                {"row": 2, "col": 2},
                {"row": 1, "col": 3},
                {"row": 2, "col": 3},
            ],
        )

        img_bytes = fig.to_image(format="png")  # kaleido library
        im = PIL.Image.open(BytesIO(img_bytes))
        pred_img = wandb.Image(im)

        y_hats[idx]
        y_hat_cum = y_hats_cum_sum[idx]
        y_close_cum_row = y_close_cum_sum[idx]
        wandb.Image(im)
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
        wandb.Image(im)
        base = 0
        y_max = torch.max(y_close_cum_row) - base
        y_min = torch.min(y_close_cum_row) - base
        # y_max = np.max(y_close_cum_sum)
        # y_min = np.min(y_close_cum_sum)
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


def add_market_viz(fig, pred_input):
    train_data_rows = pred_input.train_data_rows
    prediction_date_time = pred_input.prediction_date_time
    fig.add_trace(
        go.Ohlc(
            x=train_data_rows["time"],
            open=train_data_rows["open"],
            high=train_data_rows["high"],
            low=train_data_rows["low"],
            close=train_data_rows["close"],
        ),
        row=1,
        col=2,
    )
    # add a bar at prediction time
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(title=prediction_date_time, font=dict(size=20))
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ],
    )


def add_model_prediction(fig, pl_module, pred_input, pred_output):
    fig.update_layout(title=pred_input.prediction_date_time, font=dict(size=20))
    pl_module.plot_prediction(
        pred_input.x,
        pred_output.out,
        idx=pred_input.idx,
        ax=fig,
        row=1,
        col=1,
        draw_mode="pred_cum",
        x_time=pred_input.x_time,
    )


def add_model_interpretation(fig, model, pred_input, pred_output):
    interpretation = {}
    interp_output =  pred_output.interp_output
    for name in interp_output.keys():
        if interp_output[name].dim() > 1:
            interpretation[name] = interp_output[name][pred_input.idx]
        else:
            interpretation[name] = interp_output[name]
    model.plot_interpretation(
        interpretation,
        ax=fig,
        cells=[
            {"row": 1, "col": 2},
            {"row": 2, "col": 2},
            {"row": 1, "col": 3},
            {"row": 2, "col": 3},
        ],
    )


# @profile
def create_viz_row(
    idx,
    y_hats,
    y_hats_cum,
    y_close,
    y_close_cum_sum,
    indices,
    matched_eval_data,
    x,
    config,
    pl_module,
    out,
    target_size,
    interp_output,
    rmse,
    mae,
    filter_small=True,
    show_viz=True,
):
    y_hats[idx]
    y_hat_cum = y_hats_cum[idx]
    y_hat_cum_max = torch.max(y_hat_cum)
    y_hat_cum_min = torch.min(y_hat_cum)
    index = indices.iloc[idx]
    # logging.info(f"index:{index}")
    train_data_row = matched_eval_data[
        matched_eval_data.time_idx == index.time_idx
    ].iloc[0]
    # logging.info(f"train_data_row:{train_data_row}")
    dm = train_data_row["time"]
    dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
    y_close_cum_sum_row = y_close_cum_sum[idx]
    y_close[idx]
    y_close_cum_max = torch.max(y_close_cum_sum_row)
    y_close_cum_min = torch.min(y_close_cum_sum_row)
    if filter_small and not (
        abs(y_hat_cum_max) > 0.01
        or abs(y_hat_cum_min) > 0.01
        or abs(y_close_cum_max) > 0.01
        or abs(y_close_cum_min) > 0.01
    ):
        return {}
    train_data_rows = matched_eval_data[
        (matched_eval_data.time_idx >= index.time_idx - config.model.context_length)
        & (matched_eval_data.time_idx < index.time_idx + config.model.prediction_length)
    ]
    # logging.info(f"train_data:rows:{train_data_rows}")
    if target_size > 1:
        context_length = len(x["encoder_target"][0][idx])
    else:
        context_length = len(x["encoder_target"][idx])
    prediction_length = len(x["decoder_time_idx"][idx])
    decoder_time_idx = x["decoder_time_idx"][idx][0].cpu().detach().numpy()
    x_time = matched_eval_data[
        (matched_eval_data.time_idx >= decoder_time_idx - context_length)
        & (matched_eval_data.time_idx < decoder_time_idx + prediction_length)
    ]["time"]
    prediction_date_time = (
        train_data_row["ticker"]
        + " "
        + dm_str
        + " "
        + day_of_week_map[train_data_row["day_of_week"]]
        + " "
        + str(train_data_row["close"])
    )
    fig = None
    raw_im = None
    img = None
    if show_viz:
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
        fig.update_layout(title=prediction_date_time, font=dict(size=20))
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                # dict(
                #    bounds=[17, 2], pattern="hour"
                # ),  # hide hours outside of 4am-5pm
            ],
        )
        img_bytes = fig.to_image(format="png")  # kaleido library
        raw_im = PIL.Image.open(BytesIO(img_bytes))

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
            ],
        )
        fig.update_layout(
            autosize=False,
            width=1500,
            height=800,
            yaxis=dict(
                side="right",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                # dict(
                #    bounds=[17, 4], pattern="hour"
                # ),  # hide hours outside of 4am-5pm
            ],
        )
        fig.update_layout(title=prediction_date_time, font=dict(size=20))
        pl_module.plot_prediction(
            x,
            out,
            idx=idx,
            ax=fig,
            row=1,
            col=1,
            draw_mode="pred_cum",
            x_time=x_time,
        )
        pl_module.plot_prediction(
            x,
            out,
            idx=idx,
            ax=fig,
            row=2,
            col=1,
            draw_mode="pred_pos",
            x_time=x_time,
        )
        interpretation = {}
        for name in interp_output.keys():
            if interp_output[name].dim() > 1:
                interpretation[name] = interp_output[name][idx]
            else:
                interpretation[name] = interp_output[name]
        pl_module.plot_interpretation(
            interpretation,
            ax=fig,
            cells=[
                {"row": 1, "col": 2},
                {"row": 2, "col": 2},
                {"row": 1, "col": 3},
                {"row": 2, "col": 3},
            ],
        )
        img_bytes = fig.to_image(format="png")  # kaleido library
        im = PIL.Image.open(BytesIO(img_bytes))
        img = wandb.Image(im)

    row = {
        "ticker": train_data_row["ticker"],  # 0 ticker
        "dm": dm,  # 1 time
        "time_idx": train_data_row["time_idx"],  # 2 time_idx
        "day_of_week": train_data_row["day_of_week"],  # 3 day of week
        "hour_of_day": train_data_row["hour_of_day"],  # 4 hour of day
        "year": train_data_row["year"],  # 5 year
        "month": train_data_row["month"],  # 6 month
        "day_of_month": train_data_row["day_of_month"],  # 7 day_of_month
        "image": wandb.Image(raw_im),  # 8 image
        "y_close_cum_max": y_close_cum_max,  # 9 max
        "y_close_cum_min": y_close_cum_min,  # 10 min
        "close_back_cumsum": 0,  # 11 close_back_cusum
        "dm_str": dm_str,  # 12
        "decoder_time_idx": decoder_time_idx,
        "y_hat_cum_max": y_hat_cum_max,
        "y_hat_cum_min": y_hat_cum_min,
        "pred_img": img,
        "error_cum_max": y_hat_cum_max - y_close_cum_max,
        "error_cum_min": y_hat_cum_min - y_close_cum_min,
        "rmse": rmse[idx],
        "mae": mae[idx],
    }
    return row
