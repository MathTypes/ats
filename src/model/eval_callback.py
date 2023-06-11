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
        self, data_module, num_samples=1000
    ):
        super().__init__(["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "act_close_pct_max", "act_close_pct_min"],
                         ["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "act_close_pct_max", "act_close_pct_min",
                          "pred_time_idx", "pred_close_pct_max", "pred_close_pct_min", "img", "error_max", "error_min"])
        self.val_x, self.val_y = next(iter(data_module.val_dataloader()))
        logging.info(f"self.val_x:{self.val_x}")
        logging.info(f"self.val_y:{self.val_y}")
        self.indices = data_module.training.x_to_index(self.val_x)
        logging.info(f"self.indices:{self.indices}")
        train_data = data_module.train_data
        #self.matched_train_data = train_data[train_data.isin({"time_idx":self.indices})]
        self.matched_train_data = train_data
        logging.info(f"self.matched_train_data:{self.matched_train_data}")
        self.ticker_decoder = data_module.training.categorical_encoders["ticker"]
        self.num_samples = num_samples

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

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.trainer = trainer        
        self.pl_module = pl_module
        logging.info(f"pl_module:{self.pl_module}, {dir(self.pl_module)}")
        super().on_train_epoch_end(trainer, pl_module)
        super().on_epoch_end(trainer.current_epoch)
        
    def add_ground_truth(self, logs=None):
        y_close_cum_sum = torch.cumsum(self.val_y[0], dim=-1)
        #logging.info(f"y_close_cum_sum:{y_close_cum_sum}")
        for idx in range(self.num_samples):
            #logging.info(f"idx:{idx}, y_close_cum_sum:{y_close_cum_sum}")
            index = self.indices.iloc[idx]
            #logging.info(f"index:{index}")
            train_data_row = self.matched_train_data[self.matched_train_data.time_idx == index.time_idx].iloc[0]
            #logging.info(f"train_data_row:{train_data_row}")
            dm = train_data_row["time"]
            #logging.info(f"dm:{dir(dm)}")
            y_close_cum_sum_row = y_close_cum_sum[idx]
            #logging.info(f"y_close_cum_sum_row:{y_close_cum_sum_row}")
            self.data_table.add_data(
                train_data_row["ticker"], # ticker
                dm, # time
                train_data_row["time_idx"], # time_idx
                train_data_row["day_of_week"], # day of week
                train_data_row["hour_of_day"], # hour of day
                #wandb.Image(image),
                #np.argmax(label, axis=-1)
                torch.max(y_close_cum_sum_row),
                torch.min(y_close_cum_sum_row)
            )
        logging.info(f"self.data_table:{self.data_table}")

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()
        logging.info(f"preds:{preds}")
        logging.info(f"table_idxs:{table_idxs}")
        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                #epoch,
                self.data_table_ref.data[idx][0], # ticker
                self.data_table_ref.data[idx][1], # time
                self.data_table_ref.data[idx][2], # time_idx
                self.data_table_ref.data[idx][3], # day of week
                self.data_table_ref.data[idx][4], # hour of day
                self.data_table_ref.data[idx][5], # act_max_close_pct
                self.data_table_ref.data[idx][6], # act_min_close_pct
                pred[0], # pred_time_idx
                pred[1], # pred_close_pct_max
                pred[2], # pred_close_pct_min
                pred[3], # img
                pred[1] - self.data_table_ref.data[idx][5], # error_max
                pred[2] - self.data_table_ref.data[idx][6], # error_min
                
            )

    def _inference(self):
      preds = []
      x = self.val_x
      y = self.val_y
      
      #logging.info(f"x:{x}")
      #logging.info(f"y:{y}")
      device = self.pl_module.device
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
      #logging.info(f"y_hats:{y_hats.shape}")
      #logging.info(f"y_quantiles:{y_quantiles.shape}")
      for idx in range(self.num_samples):
          fig = self.pl_module.plot_prediction(x, out, idx=idx, add_loss_to_title=True)
          img_bytes = fig.to_image(format="png") # kaleido library
          im = PIL.Image.open(BytesIO(img_bytes))
          decoder_time_idx = x["decoder_time_idx"][idx][-1]
          #logging.info(f"decoder_time_idx:{decoder_time_idx}")
          y_hat = y_hats[idx]
          y_raw = y_raws[idx]
          #logging.info(f"y_hat:{y_hat}")
          #logging.info(f"y_raw:{y_raw}")
          y_hat_cum_sum = torch.cumsum(y_hat, dim=-1)
          #y_raw_cum_sum = torch.cumsum(y_raw, dim=-1)
          img = wandb.Image(im)
          preds.append([decoder_time_idx, torch.max(y_hat_cum_sum), torch.min(y_hat_cum_sum), img])
      #logging.info(f"preds:{len(preds)}")
      return preds
     
