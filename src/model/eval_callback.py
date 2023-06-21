from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
import logging
import PIL
import pandas as pd
from typing import Any, Dict, Optional, Set
import torch
from wandb.keras import WandbEvalCallback
import wandb
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_forecasting.utils import create_mask, detach, to_list

class WandbClfEvalCallback(WandbEvalCallback, Callback):
    def __init__(
            self, data_module, num_samples=100, every_n_epochs=10
    ):
        super().__init__(["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "year", "month", "day_of_month",
                          "act_close_pct_max", "act_close_pct_min", "close_back_cumsum"],
                         ["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "year", "month", "day_of_month",
                          "act_close_pct_max", "act_close_pct_min", "close_back_cumsum",
                          "pred_time_idx", "pred_close_pct_max", "pred_close_pct_min", "img", "error_max", "error_min", "img_interp"])
        self.val_x_batch = []
        self.val_y_batch = []
        self.indices_batch = []
        for batch in range(num_samples):
            val_x, val_y = next(iter(data_module.val_dataloader()))
            #logging.info(f"self.val_x:{val_x}")
            #logging.info(f"self.val_y:{val_y}")
            indices = data_module.validation.x_to_index(val_x)
            self.val_x_batch.append(val_x)
            self.val_y_batch.append(val_y)
            self.indices_batch.append(indices)
        #logging.info(f"self.indices:{self.indices}")
        eval_data = data_module.eval_data
        #self.matched_train_data = train_data[train_data.isin({"time_idx":self.indices})]
        self.matched_eval_data = eval_data
        #logging.info(f"self.matched_train_data:{self.matched_train_data}")
        #self.ticker_decoder = data_module.training.categorical_encoders["ticker"]
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

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
        #y_close_cum_sum = torch.cumsum(self.val_y[0], dim=-1)
        #logging.info(f"y_close_cum_sum:{y_close_cum_sum}")
        #logging.info(f"val_x:{self.val_x}")
        for batch_idx in range(self.num_samples):
            val_x = self.val_x_batch[batch_idx]
            val_y = self.val_y_batch[batch_idx]
            indices = self.indices_batch[batch_idx]
            y_close_cum_sum = val_y[0]
            #logging.info(f"y_close_cum_sum:{y_close_cum_sum}, len:{len(y_close_cum_sum)}, shape:{y_close_cum_sum.shape}")
            for idx in range(len(y_close_cum_sum)):
              base = val_x['encoder_target'][idx][-1]
              #logging.info(f"idx:{idx}, y_close_cum_sum:{y_close_cum_sum}")
              #logging.info(f"encoder_x:{self.val_x['encoder_x'][idx]}")
              #logging.info(f"encoder_target:{self.val_x['encoder_target'][idx]}")
              #logging.info(f"decoder_target:{self.val_x['decoder_target'][idx]}")
              index = indices.iloc[idx]
              #logging.info(f"index:{index}")
              train_data_row = self.matched_eval_data[self.matched_eval_data.time_idx == index.time_idx].iloc[0]
              #logging.info(f"train_data_row:{train_data_row}")
              dm = train_data_row["time"]
              #logging.info(f"dm:{dir(dm)}")
              y_close_cum_sum_row = y_close_cum_sum[idx]
              #logging.info(f"y_close_cum_sum_row:{y_close_cum_sum_row}")
              self.data_table.add_data(
                train_data_row["ticker"], # 0 ticker
                dm, # 1 time
                train_data_row["time_idx"], # 2 time_idx
                train_data_row["day_of_week"], # 3 day of week
                train_data_row["hour_of_day"], # 4 hour of day
                train_data_row["year"], # 5 year
                train_data_row["month"], # 6 month
                train_data_row["day_of_month"], # 7 day_of_month
                #wandb.Image(image),
                #np.argmax(label, axis=-1)
                torch.max(y_close_cum_sum_row) - base, # 8 max
                torch.min(y_close_cum_sum_row) - base, # 9 min
                base, # 10 close_back_cusum
              )
        #logging.info(f"self.data_table:{self.data_table}")

    def add_model_predictions(self, epoch, logs=None):
        if epoch % self.every_n_epochs:
            return
        
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()
        #logging.info(f"preds:{preds}")
        #logging.info(f"table_idxs:{table_idxs}")
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
                self.data_table_ref.data[idx][8], # act_max_close_pct
                self.data_table_ref.data[idx][9], # act_min_close_pct
                self.data_table_ref.data[idx][10], # close_back_cumsum
                pred[0], # pred_time_idx
                pred[1] - self.data_table_ref.data[idx][10], # pred_close_pct_max
                pred[2] - self.data_table_ref.data[idx][10], # pred_close_pct_min
                pred[3], # img
                pred[1] - self.data_table_ref.data[idx][10] - self.data_table_ref.data[idx][8], # error_max
                pred[2] - self.data_table_ref.data[idx][10] - self.data_table_ref.data[idx][9], # error_min
                pred[4], # img_interp
            )

    def _inference(self):
      preds = []
      device = self.pl_module.device
      for batch_idx in range(self.num_samples):
        x = self.val_x_batch[batch_idx]
        y = self.val_y_batch[batch_idx]
      
        #logging.info(f"x:{x}")
        #logging.info(f"inference val_y:{y}")
        x = {key:val.to(device) for key, val in x.items()}
        y = [val.to(device) if val is not None else None for val in y]
        kwargs={'nolog': True}
        #logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        log, out = self.pl_module.step(x=x, y=y, batch_idx=0, **kwargs)
        #logging.info(f"log:{log}")
        #logging.info(f"out:{out}")

        y_raws = to_list(out["prediction"])[0]  # raw predictions - used for calculating loss
        prediction_kwargs = {}
        quantiles_kwargs = {}
        y_hats = to_list(self.pl_module.to_prediction(out, **prediction_kwargs))[0]
        y_quantiles = to_list(self.pl_module.to_quantiles(out, **quantiles_kwargs))[0]
        #logging.info(f"y_raws:{y_raws.shape}")
        #logging.info(f"y_hats:{y_hats}")
        #exit(0)
        #logging.info(f"y_quantiles:{y_quantiles.shape}")
        for idx in range(len(y_hats)):
          prediction_date_time = str(self.data_table_ref.data[idx][5]) + "-" + str(self.data_table_ref.data[idx][6]) + "-" + str(self.data_table_ref.data[idx][7]) + " " + str(self.data_table_ref.data[idx][4])
          fig = self.pl_module.plot_prediction(x, out, idx=idx, add_loss_to_title=True)
          fig.update_layout(title=prediction_date_time)
          img_bytes = fig.to_image(format="png") # kaleido library
          im = PIL.Image.open(BytesIO(img_bytes))
          decoder_time_idx = x["decoder_time_idx"][idx][-1]
          #logging.info(f"decoder_time_idx:{decoder_time_idx}")
          y_hat = y_hats[idx]
          y_raw = y_raws[idx]
          #logging.info(f"y_hat:{y_hat}")
          #logging.info(f"y_raw:{y_raw}")
          #y_hat_cum_sum = torch.cumsum(y_hat, dim=-1)
          #y_raw_cum_sum = torch.cumsum(y_raw, dim=-1)
          img = wandb.Image(im)

          fig_interp = self.pl_module.plot_interpretation(x, out, idx=idx)
          fig.update_layout(title=prediction_date_time)
          img_bytes_interp = fig_interp.to_image(format="png") # kaleido library
          im_interp = PIL.Image.open(BytesIO(img_bytes_interp))
          img_interp = wandb.Image(im_interp)

          preds.append([decoder_time_idx, torch.max(y_hat), torch.min(y_hat), img, img_interp])
      #logging.info(f"preds:{len(preds)}")
      return preds
     
