import logging
import pandas as pd
from typing import Any, Dict, Optional, Set
from wandb.keras import WandbEvalCallback
import wandb
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

class WandbClfEvalCallback(WandbEvalCallback, Callback):
    def __init__(
        self, data_module, num_samples=10000
    ):
        super().__init__(["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "max_act_close_pct", "min_act_close_pct"],
                         ["ticker", "time", "time_idx", "day_of_week", "hour_of_day", "max_act_close_pct", "min_act_close_pct",
                          "pred_time_idx", "pred_max_act_close", "pred_min_act_close", "img"])
        self.val_data = next(iter(data_module.val_dataloader()))
        self.X_val = data_module.X_val
        self.y_val = data_module.y_val
        self.num_samples = num_samples
        logging.info(f"self.val_data:{len(self.val_data)}")
        logging.info(f"self.X_val:{len(self.X_val)}")
        logging.info(f"self.y_val:{len(self.y_val)}")

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
        super().on_train_epoch_end(trainer, pl_module)
        super().on_epoch_end(trainer.current_epoch)
        
    def add_ground_truth(self, logs=None):        
        for idx in range(self.num_samples):
            y_close_cum_sum = self.y_val[idx][4].cumsum()
            #logging.info(f"idx:{idx}, y_close_cum_sum:{y_close_cum_sum}")
            dm = pd.to_datetime(self.X_val[idx][1])
            self.data_table.add_data(
                self.X_val[idx][0], # ticker
                dm, # time
                self.X_val[idx][2], # time_idx
                dm.isocalendar().day, # day of week
                self.X_val[idx][3], # hour of day
                #wandb.Image(image),
                #np.argmax(label, axis=-1)
                y_close_cum_sum.max(),
                y_close_cum_sum.min()
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
                epoch,
                self.data_table_ref.data[idx][0], # ticker
                self.data_table_ref.data[idx][1], # time
                self.data_table_ref.data[idx][2], # time_idx
                self.data_table_ref.data[idx][3], # day of week
                self.data_table_ref.data[idx][4], # hour of day
                self.data_table_ref.data[idx][5], # act_max_close_pct
                self.data_table_ref.data[idx][6], # act_min_close_pct
                pred[0],
                pred[1]
            )

    def _inference(self):
      preds = []
      x, y = self.val_data
      
      logging.info(f"x:{x}")
      logging.info(f"y:{y}")
      device = self.pl_module.device
      x = {key:val.to(device) for key, val in x.items()}
      y = [val.to(device) if val is not None else None for val in y]
      log, out = self.pl_module.step(x, y, 0, nolog=True)
      logging.info(f"log:{log}")
      logging.info(f"out:{out}")

      y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
      prediction_kwargs = {}
      quantiles_kwargs = {}
      y_hats = to_list(self.pl_module.to_prediction(out, **prediction_kwargs))
      y_quantiles = to_list(self.pl_module.to_quantiles(out, **quantiles_kwargs))
      logging.info(f"y_raws:{y_raws}")
      logging.info(f"y_hats:{y_hats}")
      logging.info(f"y_quantiles:{y_quantiles}")
      for idx in range(self.num_samples):
          fig = self.plot_prediction(x, out, idx=idx, add_loss_to_title=True)
          img_bytes = fig.to_image(format="png") # kaleido library
          im = PIL.Image.open(BytesIO(img_bytes))
          time_idx = x["time_idx"][-1]
          y_hat = y_hats[idx]
          y_hat_cum_sum = y_hat.cumsum()
          img = wandb.Image(im)
          preds.append([time_idx, y_hat_cum_sum.max(), y_hat_cum_sum.min(), img])
      logging.info(f"preds:{len(preds)}")
      return preds
     
