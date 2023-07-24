import gc
import logging
from typing import Any, List

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_forecasting.utils import detach, to_list
import torch
import wandb
from wandb.keras import WandbEvalCallback

from ats.model import viz_utils

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
                "close_back",
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
            logging.info(
                f"batch_size:{len(val_x)}, indices_batch:{len(self.indices_batch)}"
            )
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
            "close_back",
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
        device = self.pl_module.device
        for batch_idx in range(self.num_samples):
            logging.info(f"add prediction for batch:{batch_idx}")
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"allocated before prediction:{torch.cuda.memory_allocated()}")
            logging.info(f"max_allocated before prediction:{torch.cuda.max_memory_allocated()}")
            orig_x = self.val_x_batch[batch_idx]
            orig_y = self.val_y_batch[batch_idx]
            indices = self.indices_batch[batch_idx]
            y_close = orig_y[0]
            # TODO: fix following hack to deal with multiple targets
            if isinstance(y_close, list):
                y_close = y_close[0]
            y_close_cum_sum = torch.cumsum(y_close, dim=-1)
            x = {
                key: [v.to(device) for v in val]
                if isinstance(val, list)
                else val.to(device)
                for key, val in orig_x.items()
            }
            y = [
                [v.to(device) for v in val]
                if isinstance(val, list)
                else val.to(device)
                if val is not None
                else None
                for val in orig_y
            ]
            kwargs = {"nolog": True}
            # logging.info(f"x:{x.device}")
            # logging.info(f"y:{y.device}")
            # logging.info(f"self.pl_module:{self.pl_module.device}")
            log, out = self.pl_module.step(x=x, y=y, batch_idx=0, **kwargs)
            logs = detach(log)
            # logging.info(f"out:{out}")
            prediction_kwargs = {"reduction": None}
            result = self.pl_module.compute_metrics(
                x, y, out, prediction_kwargs=prediction_kwargs
            )
            result = {k:detach(v) for k, v in result.items()}
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
            predictions = detach(predictions)
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
            logging.info(f"allocated before iter:{torch.cuda.memory_allocated()}")
            logging.info(f"max_allocated before iter:{torch.cuda.max_memory_allocated()}")
            for idx in range(len(y_hats)):
                #logging.info(f"allocated:{torch.cuda.memory_allocated()}")
                #logging.info(f"max_allocated:{torch.cuda.max_memory_allocated()}")
                row = viz_utils.create_viz_row(
                    idx,
                    y_hats,
                    y_hats_cum,
                    y_close,
                    y_close_cum_sum,
                    indices,
                    self.matched_eval_data,
                    x,
                    self.config,
                    self.pl_module,
                    out,
                    self.target_size,
                    interp_output,
                    rmse,
                    mae,
                )
                if row:
                    data_table.add_data(
                        row["ticker"],  # 0 ticker
                        row["dm"],  # 1 time
                        row["time_idx"],  # 2 time_idx
                        row["day_of_week"],  # 3 day of week
                        row["hour_of_day"],  # 4 hour of day
                        row["year"],  # 5 year
                        row["month"],  # 6 month
                        row["day_of_month"],  # 7 day_of_month
                        row["image"],  # 8 image
                        row["y_close_cum_max"],  # 9 max
                        row["y_close_cum_min"],  # 10 min
                        row["close_back_cumsum"],  # 11 close_back_cusum
                        row["close_back"],
                        row["dm_str"],  # 12
                        row["decoder_time_idx"],
                        row["y_hat_cum_max"],
                        row["y_hat_cum_min"],
                        row["pred_img"],
                        row["error_cum_max"],
                        row["error_cum_min"],
                        row["rmse"],
                        row["mae"],
                    )
                    cnt += 1
            del x
            del y
            del y_raws
            del interp_output
            del predictions
            del y_hats_cum
            del y_quantiles
            del y_close_cum_sum
            del log
            del out
            del y_hats
            del result
            torch.cuda.empty_cache()
            logging.info(f"added {cnt} examples")
        # logging.info(f"preds:{len(preds)}")
        data_artifact.add(data_table, "eval_data")
        # Calling `use_artifact` uploads the data to W&B.
        assert wandb.run is not None
        wandb.run.use_artifact(data_artifact)
        del data_table
        # data_artifact.wait()

        return []
