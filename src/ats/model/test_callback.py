from io import BytesIO
import logging
from omegaconf import OmegaConf
import PIL
import pandas as pd
import torch
from wandb.keras import WandbEvalCallback
import wandb
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_forecasting.utils import create_mask, detach, to_list


class TestEvalCallback(Callback):
    def __init__(self, data_module, target, num_samples=10, every_n_epochs=5):
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
                "act_close_pct_max",
                "act_close_pct_min",
                "close_back_cumsum",
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
                "act_close_pct_max",
                "act_close_pct_min",
                "close_back_cumsum",
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
        self.target = target
        self.target_size = len(self.target)
        for val_x, val_y in iter(data_module.test_dataloader()):
            indices = data_module.validation.x_to_index(val_x)
            self.val_x_batch.append(val_x)
            self.val_y_batch.append(val_y)
            self.indices_batch.append(indices)
        eval_data = data_module.test_data
        self.matched_eval_data = eval_data
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.trainer = trainer
        self.pl_module = pl_module
        logging.info(f"trainer:current_epoch: {trainer.current_epoch}")
        if trainer.current_epoch % self.every_n_epochs:
            return
        self.pl_module.device
        for batch_idx in range(self.num_samples):
            x = self.val_x_batch[batch_idx]
            y = self.val_y_batch[batch_idx]

            kwargs = {"nolog": True}
            log, out = self.pl_module.step(x=x, y=y, batch_idx=0, **kwargs)
            prediction_kwargs = {"reduction": None}
            result = self.pl_module.compute_metrics(
                x, y, out, prediction_kwargs=prediction_kwargs
            )
            result["train_RMSE"].cpu().detach().numpy()
            result["train_MAE"].cpu().detach().numpy()
            y_raws = to_list(out["prediction"])[
                0
            ]  # raw predictions - used for calculating loss
            prediction_kwargs = {}
            quantiles_kwargs = {}
            y_hats = to_list(self.pl_module.to_prediction(out, **prediction_kwargs))[0]
            y_quantiles = to_list(self.pl_module.to_quantiles(out, **quantiles_kwargs))[
                0
            ]
            for idx in range(len(y_hats)):
                prediction_date_time = (
                    str(self.data_table_ref.data[idx][5])
                    + "-"
                    + str(self.data_table_ref.data[idx][6])
                    + "-"
                    + str(self.data_table_ref.data[idx][7])
                    + " "
                    + str(self.data_table_ref.data[idx][3])
                    + " "
                    + str(self.data_table_ref.data[idx][4])
                )
                fig = self.pl_module.plot_prediction(
                    x, out, idx=idx, add_loss_to_title=True
                )
                if isinstance(fig, (list, tuple)):
                    fig = fig[0]
                fig.update_layout(title=prediction_date_time)
                img_bytes = fig.to_image(format="png")  # kaleido library
                im = PIL.Image.open(BytesIO(img_bytes))
                x["decoder_time_idx"][idx][-1]
                y_hats[idx]
                y_raws[idx]
                wandb.Image(im)
