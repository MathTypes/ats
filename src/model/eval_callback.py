import datetime
from io import BytesIO
import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import PIL
import plotly.graph_objects as go
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from plotly.subplots import make_subplots
from pytorch_forecasting.utils import create_mask, detach, to_list
import torch
import wandb
from wandb.keras import WandbEvalCallback


day_of_week_map = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]


def get_output_by_idx(out, idx):
    return {name: out[name][idx] for name in out.keys()}


class WandbClfEvalCallback(WandbEvalCallback, Callback):
    def __init__(self, data_module, target, config):
        super().__init__(
            [
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
            ],
            [
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
            ],
        )
        self.val_x_batch = []
        self.val_y_batch = []
        self.indices_batch = []
        self.config = config
        self.target_size = len(target) if isinstance(target, List) else 1
        self.num_samples = config.job.eval_batches
        self.every_n_epochs = config.job.log_example_eval_every_n_epochs
        logging.info(f"num_samples:{self.num_samples}")
        data_iter = iter(data_module.val_dataloader())
        for batch in range(self.num_samples):
            val_x, val_y = next(data_iter)
            indices = data_module.validation.x_to_index(val_x)
            self.val_x_batch.append(val_x)
            self.val_y_batch.append(val_y)
            self.indices_batch.append(indices)
            logging.info(f"batch_size:{len(val_x)}, indices_batch:{len(self.indices_batch)}")
        self.validation = data_module.validation
        self.matched_eval_data = data_module.eval_data
        self.returns_target_name = self.validation.target_names[0]
        transformer = self.validation.get_transformer(self.returns_target_name)
        logging.info(f"transformer:{transformer}")


    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # super().on_train_end(trainer, pl_module)
        super().on_train_end(logs=None)

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_start(trainer, pl_module)
        super().on_train_begin()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        super().on_train_batch_end(batch)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        super().on_train_batch_end(batch)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.trainer = trainer
        self.pl_module = pl_module
        # logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        super().on_train_epoch_end(trainer, pl_module)
        super().on_epoch_end(trainer.current_epoch)

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.trainer = trainer
        self.pl_module = pl_module
        # logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        super().on_test_epoch_end(trainer, pl_module)

    def add_ground_truth(self, logs=None):
        for batch_idx in range(self.num_samples):
            pass
        
    def add_model_predictions(self, epoch, logs=None):
        if epoch % self.every_n_epochs:
            return
        data_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred_viz", type="pred_viz")
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
        preds = []
        device = self.pl_module.device
        for batch_idx in range(self.num_samples):
            logging.info(f"add prediction for batch:{batch_idx}")
            x = self.val_x_batch[batch_idx]
            y = self.val_y_batch[batch_idx]
            indices = self.indices_batch[batch_idx]
            y_close = y[0]
            # TODO: fix following hack to deal with multiple targets                                                                                     
            # logging.info(f"y_close_cum_sum:{type(y_close_cum_sum)}")                                                                                   
            if isinstance(y_close, list):
                y_close = y_close[0]
            y_close_cum_sum = torch.cumsum(y_close, dim=-1)
            x = {
                key: [v.to(device) for v in val]
                if isinstance(val, list)
                else val.to(device)
                for key, val in x.items()
            }
            y = [
                [v.to(device) for v in val]
                if isinstance(val, list)
                else val.to(device)
                if val is not None
                else None
                for val in y
            ]
            kwargs = {"nolog": True}
            #logging.info(f"x:{x.device}")
            #logging.info(f"y:{y.device}")
            #logging.info(f"self.pl_module:{self.pl_module.device}")
            log, out = self.pl_module.step(x=x, y=y, batch_idx=0, **kwargs)
            # logging.info(f"out:{out}")
            prediction_kwargs = {"reduction": None}
            result = self.pl_module.compute_metrics(
                x, y, out, prediction_kwargs=prediction_kwargs
            )
            if "train_RMSE" in result:
                rmse = result["train_RMSE"].cpu().detach().numpy()
            else:
                rmse = result["close_back_cumsum train_RMSE"].cpu().detach().numpy()
            if "train_MAE" in result:
                mae = result["train_MAE"].cpu().detach().numpy()
            else:
                mae = result["close_back_cumsum train_MAE"].cpu().detach().numpy()
            y_raws = to_list(out["prediction"])[
                0
            ]  # raw predictions - used for calculating loss
            prediction_kwargs = {}
            quantiles_kwargs = {}
            predictions = self.pl_module.to_prediction(out, **prediction_kwargs)
            # logging.info(f"predictions:{predictions}")
            y_hats = to_list(predictions)[0]
            y_hats_cum = torch.cumsum(y_hats, dim=-1)
            y_quantiles = to_list(self.pl_module.to_quantiles(out, **quantiles_kwargs))[
                0
            ]
            interp_output = self.pl_module.interpret_output(
                detach(out),
                reduction="none",
                attention_prediction_horizon=0,  # attention only for first prediction horizon
            )
            cnt = 0
            for idx in range(len(y_hats)):
                y_hat = y_hats[idx]
                y_hat_cum = y_hats_cum[idx]
                y_hat_cum_max = torch.max(y_hat_cum)
                y_hat_cum_min = torch.min(y_hat_cum)
                index = indices.iloc[idx]
                #logging.info(f"index:{index}")
                train_data_row = self.matched_eval_data[
                    self.matched_eval_data.time_idx == index.time_idx
                ].iloc[0]
                #logging.info(f"train_data_row:{train_data_row}")
                dm = train_data_row["time"]
                dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
                y_close_cum_sum_row = y_close_cum_sum[idx]
                y_close_row = y_close[idx]
                y_close_cum_max = torch.max(y_close_cum_sum_row)
                y_close_cum_min = torch.min(y_close_cum_sum_row)
                if not (abs(y_hat_cum_max) > 0.01 or abs(y_hat_cum_min) > 0.01 or
                        abs(y_close_cum_max) > 0.01 or abs(y_close_cum_min) > 0.01):
                    continue
                cnt += 1
                train_data_rows = self.matched_eval_data[
                    (
                        self.matched_eval_data.time_idx
                        >= index.time_idx - self.config.model.context_length
                    )
                    & (
                        self.matched_eval_data.time_idx
                        < index.time_idx + self.config.model.prediction_length
                    )
                ]
                #logging.info(f"train_data:rows:{train_data_rows}")
                if self.target_size > 1:
                    context_length = len(x["encoder_target"][0][idx])
                else:
                    context_length = len(x["encoder_target"][idx])
                prediction_length = len(x["decoder_time_idx"][idx])
                decoder_time_idx = x["decoder_time_idx"][idx][0].cpu().detach().numpy()
                x_time = self.matched_eval_data[
                    (
                        self.matched_eval_data.time_idx
                        >= decoder_time_idx - context_length
                    )
                    & (
                        self.matched_eval_data.time_idx
                        < decoder_time_idx + prediction_length
                    )
                ]["time"]
                # logging.info(f"x_time:{x_time}")
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
                    + " " + str(train_data_row["close"])
                )
                fig.update_layout(title=prediction_date_time, font=dict(size=20))
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # hide weekends
                        dict(
                            bounds=[17, 2], pattern="hour"
                        ),  # hide hours outside of 4am-5pm
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
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # hide weekends
                        dict(
                            bounds=[17, 4], pattern="hour"
                        ),  # hide hours outside of 4am-5pm
                    ],
                )
                prediction_date_time = (
                    train_data_row["ticker"] + " " + dm_str + " "
                    + day_of_week_map[train_data_row["day_of_week"]]
                )
                fig.update_layout(title=prediction_date_time, font=dict(size=20))
                self.pl_module.plot_prediction(
                    x,
                    out,
                    idx=idx,
                    ax=fig,
                    row=1,
                    col=1,
                    draw_mode="pred_cum",
                    x_time=x_time,
                )
                self.pl_module.plot_prediction(
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
                self.pl_module.plot_interpretation(
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
                data_table.add_data(
                    train_data_row["ticker"],  # 0 ticker
                    dm,  # 1 time
                    train_data_row["time_idx"],  # 2 time_idx
                    train_data_row["day_of_week"],  # 3 day of week
                    train_data_row["hour_of_day"],  # 4 hour of day
                    train_data_row["year"],  # 5 year
                    train_data_row["month"],  # 6 month
                    train_data_row["day_of_month"],  # 7 day_of_month
                    wandb.Image(raw_im),  # 8 image
                    # np.argmax(label, axis=-1)
                    y_close_cum_max,  # 9 max
                    y_close_cum_min,  # 10 min
                    0,  # 11 close_back_cusum
                    dm_str,  # 12
                    decoder_time_idx,
                    y_hat_cum_max,
                    y_hat_cum_min,
                    img,
                    y_hat_cum_max-y_close_cum_max,
                    y_hat_cum_min-y_close_cum_min,
                    rmse[idx],
                    mae[idx],
                )
            logging.info(f"added {cnt} examples")
        # logging.info(f"preds:{len(preds)}")
        data_artifact.add(data_table, "eval_data")
        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        #data_artifact.wait()
        
        return []
