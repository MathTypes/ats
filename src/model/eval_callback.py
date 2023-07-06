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
        super().__init__(["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "year", "month", "day_of_month", "price_img",
                          "act_close_pct_max", "act_close_pct_min", "close_back_cumsum", "time_str"],
                         ["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "year", "month", "day_of_month", "price_img",
                          "act_close_pct_max", "act_close_pct_min", "close_back_cumsum", "time_str",
                          "pred_time_idx", "pred_close_pct_max", "pred_close_pct_min", "img", "error_max", "error_min", "rmse", "mae"])
        self.val_x_batch = []
        self.val_y_batch = []
        self.indices_batch = []
        self.config = config
        self.target_size = len(target) if isinstance(target, List) else 1
        self.num_samples = config.job.eval_batches
        self.every_n_epochs = config.job.log_example_eval_every_n_epochs
        for batch in range(self.num_samples):
            val_x, val_y = next(iter(data_module.val_dataloader()))
            indices = data_module.validation.x_to_index(val_x)
            self.val_x_batch.append(val_x)
            self.val_y_batch.append(val_y)
            self.indices_batch.append(indices)
        eval_data = data_module.eval_data
        self.matched_eval_data = eval_data

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #super().on_train_end(trainer, pl_module)
        super().on_train_end(logs=None)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer        
        self.pl_module = pl_module
        #logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        super().on_train_epoch_end(trainer, pl_module)
        super().on_epoch_end(trainer.current_epoch)
        
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer        
        self.pl_module = pl_module
        #logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        super().on_test_epoch_end(trainer, pl_module)

    def add_ground_truth(self, logs=None):
        for batch_idx in range(self.num_samples):
            val_x = self.val_x_batch[batch_idx]
            val_y = self.val_y_batch[batch_idx]
            indices = self.indices_batch[batch_idx]
            y_close = val_y[0]
            # TODO: fix following hack to deal with multiple targets
            #logging.info(f"y_close_cum_sum:{type(y_close_cum_sum)}")
            if isinstance(y_close, list):
                y_close = y_close[0]
            y_close_cum_sum = torch.cumsum(y_close, dim=-1)
            for idx in range(len(y_close_cum_sum)):
              # TODO: fix [0] hack to deal with multiple target
              if self.target_size > 1:
                  base = val_x['encoder_target'][0][idx][-1]
              else:
                  if val_x['encoder_target'].dim() == 2:
                    base = val_x['encoder_target'][idx][-1]
                  else:
                    base = val_x['encoder_target'][0][idx][-1]
              # when using close_back, base is always 0
              base = 0
              index = indices.iloc[idx]
              train_data_row = self.matched_eval_data[self.matched_eval_data.time_idx == index.time_idx].iloc[0]
              dm = train_data_row["time"]
              dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
              y_close_cum_sum_row = y_close_cum_sum[idx]
              train_data_rows = self.matched_eval_data[(self.matched_eval_data.time_idx>=index.time_idx-self.config.model.context_length)
                                                       & (self.matched_eval_data.time_idx<index.time_idx+self.config.model.prediction_length)]
              fig = go.Figure(data=go.Ohlc(x=train_data_rows['time'],
                    open=train_data_rows['open'],
                    high=train_data_rows['high'],
                    low=train_data_rows['low'],
                    close=train_data_rows['close']))
              # add a bar at prediction time
              fig.update(layout_xaxis_rangeslider_visible=False)
              prediction_date_time = train_data_row["ticker"] + " " + dm_str + " " + day_of_week_map[train_data_row["day_of_week"]]
              fig.update_layout(title=prediction_date_time, font=dict(size=20))
              fig.update_xaxes(
                  rangebreaks=[
                      dict(bounds=["sat", "mon"]), #hide weekends
                      dict(bounds=[17, 4], pattern="hour"), #hide hours outside of 4am-5pm
                  ],
              )
              img_bytes = fig.to_image(format="png") # kaleido library
              im = PIL.Image.open(BytesIO(img_bytes))
              if torch.max(y_close_cum_sum_row) > 0.10:
                  logging.info(f"bad row:{train_data_row}, idx:{idx}, index:{index}")
                  logging.info(f"y_close_cum_sum_row:{y_close_cum_sum_row}")
                  logging.info(f"y_close:{y_close[idx]}")
                  logging.info(f"train_data_rows:{train_data_rows[-32:]}")
              self.data_table.add_data(
                train_data_row["ticker"], # 0 ticker
                dm, # 1 time
                train_data_row["time_idx"], # 2 time_idx
                train_data_row["day_of_week"], # 3 day of week
                train_data_row["hour_of_day"], # 4 hour of day
                train_data_row["year"], # 5 year
                train_data_row["month"], # 6 month
                train_data_row["day_of_month"], # 7 day_of_month
                wandb.Image(im),  # 8 image
                #np.argmax(label, axis=-1)
                torch.max(y_close_cum_sum_row) - base, # 9 max
                torch.min(y_close_cum_sum_row) - base, # 10 min
                base, # 11 close_back_cusum
                dm_str, # 12
              )

    def add_model_predictions(self, epoch, logs=None):
        if epoch % self.every_n_epochs:
            return
        
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()
        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                #epoch,
                self.data_table_ref.data[idx][0], # ticker
                self.data_table_ref.data[idx][1], # time
                self.data_table_ref.data[idx][2], # time_idx
                self.data_table_ref.data[idx][3], # day of week
                self.data_table_ref.data[idx][4], # hour of day
                self.data_table_ref.data[idx][5], # year
                self.data_table_ref.data[idx][6], # month
                self.data_table_ref.data[idx][7], # day_of_month
                self.data_table_ref.data[idx][8], # price image
                self.data_table_ref.data[idx][9], # act_max_close_pct
                self.data_table_ref.data[idx][10], # act_min_close_pct
                self.data_table_ref.data[idx][11], # close_back_cumsum,
                self.data_table_ref.data[idx][12], # time_sr,                
                pred[0], # pred_time_idx
                pred[1] - self.data_table_ref.data[idx][10], # pred_close_pct_max
                pred[2] - self.data_table_ref.data[idx][10], # pred_close_pct_min
                pred[3], # img
                pred[1] - self.data_table_ref.data[idx][10] - self.data_table_ref.data[idx][9], # error_max
                pred[2] - self.data_table_ref.data[idx][10] - self.data_table_ref.data[idx][10], # error_min
                #pred[4], # img_interp
                pred[4], # rmse
                pred[5], # mae
            )

    def _inference(self):
      preds = []
      device = self.pl_module.device
      for batch_idx in range(self.num_samples):
        x = self.val_x_batch[batch_idx]
        y = self.val_y_batch[batch_idx]
      
        x = {key:[v.to(device) for v in val] if isinstance(val, list) else val.to(device) for key, val in x.items()}
        y = [[v.to(device) for v in val] if isinstance(val, list) else val.to(device) if val is not None else None for val in y]
        kwargs={'nolog': True}
        log, out = self.pl_module.step(x=x, y=y, batch_idx=0, **kwargs)
        prediction_kwargs = {'reduction':None}
        result = self.pl_module.compute_metrics(x, y, out, prediction_kwargs=prediction_kwargs)
        if "train_RMSE" in result:
            rmse = result["train_RMSE"].cpu().detach().numpy()
        else:
            rmse = result["close_back_cumsum train_RMSE"].cpu().detach().numpy()
        if "train_MAE" in result:
            mae = result["train_MAE"].cpu().detach().numpy()
        else:
             mae = result["close_back_cumsum train_MAE"].cpu().detach().numpy()
        y_raws = to_list(out["prediction"])[0]  # raw predictions - used for calculating loss
        prediction_kwargs = {}
        quantiles_kwargs = {}
        y_hats = to_list(self.pl_module.to_prediction(out, **prediction_kwargs))[0]
        y_hats_cum = torch.cumsum(y_hats, dim=-1)
        y_quantiles = to_list(self.pl_module.to_quantiles(out, **quantiles_kwargs))[0]
        interp_output = self.pl_module.interpret_output(
            detach(out),
            reduction="none",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        for idx in range(len(y_hats)):
          context_length = len(x["encoder_target"][idx])
          prediction_length = len(x["decoder_time_idx"][idx])
          decoder_time_idx = x["decoder_time_idx"][idx][0].cpu().detach().numpy()
          x_time = self.matched_eval_data[(self.matched_eval_data.time_idx>=decoder_time_idx-context_length) & (self.matched_eval_data.time_idx<decoder_time_idx+prediction_length)]["time"]
          #logging.info(f"x_time:{x_time}")
          fig = make_subplots(rows=2, cols=3, specs=[[{"secondary_y": True}, {"secondary_y": True},  {"secondary_y": True}],
                                                     [{"secondary_y": True}, {"secondary_y": True},  {"secondary_y": True}]], )
          fig.update_layout(
              autosize=False, width=1500, height=800,
          )
          fig.update_xaxes(
              rangebreaks=[
                  dict(bounds=["sat", "mon"]), #hide weekends
                  dict(bounds=[17, 4], pattern="hour"), #hide hours outside of 4am-5pm
              ],
          )
          prediction_date_time = self.data_table_ref.data[idx][0] + " " + str(self.data_table_ref.data[idx][12]) + " " + day_of_week_map[self.data_table_ref.data[idx][3]]
          fig.update_layout(title=prediction_date_time, font=dict(size=20))
          self.pl_module.plot_prediction(x, out, idx=idx, ax=fig, row=1, col=1, draw_mode="pred", x_time=x_time)
          self.pl_module.plot_prediction(x, out, idx=idx, ax=fig, row=2, col=1, draw_mode="pred_cum", x_time=x_time)
          interpretation = {}
          for name in interp_output.keys():
              if interp_output[name].dim()>1:
                  interpretation[name] = interp_output[name][idx]
              else:
                  interpretation[name] = interp_output[name]
          self.pl_module.plot_interpretation(interpretation, ax=fig,
                                             cells =[{"row":1, "col":2}, {"row":2, "col":2},
                                                     {"row":1, "col":3}, {"row":2, "col":3}])
          img_bytes = fig.to_image(format="png") # kaleido library
          im = PIL.Image.open(BytesIO(img_bytes))
          y_hat = y_hats[idx]
          y_hat_cum = y_hats_cum[idx]
          y_raw = y_raws[idx]
          img = wandb.Image(im)
          preds.append([decoder_time_idx, torch.max(y_hat_cum), torch.min(y_hat_cum), img, rmse[idx], mae[idx]])
      #logging.info(f"preds:{len(preds)}")
      return preds
     
