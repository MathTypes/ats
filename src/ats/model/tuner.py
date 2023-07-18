import logging
import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

from pytorch_forecasting.data.examples import get_stallion_data
from datasets import generate_stock_returns
from util import logging_utils

from math import ceil


def week_of_month(dt):
    """Returns the week of the month for the specified date."""
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom / 7.0))


if __name__ == "__main__":
    logging_utils.init_logging()

    data = pd.read_parquet("data/token/FUT/30min/ES", engine="fastparquet")
    data["Time"] = data.index
    data["ticker"] = "ES"
    data["volume"] = data["Volume"]
    data["close"] = data["Close"]
    data["Time"] = data["Time"].apply(lambda x: x.timestamp()).astype(np.float32)
    logging.info(f"data:{data.head()}")

    data["date"] = data.index
    # add time index
    data.insert(0, "time_idx", range(0, len(data)))
    # data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    # data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_ticker"] = data.groupby(
        ["time_idx", "ticker"], observed=True
    ).volume.transform("mean")
    data["hour_of_day"] = (
        data["date"].apply(lambda x: x.hour).astype(str).astype("category")
    )
    data["day_of_week"] = data.index.dayofweek.astype(str).astype("category")
    data["day_of_month"] = data.index.day.astype(str).astype("category")
    data["week_of_month"] = (
        data["date"].apply(week_of_month).astype(str).astype("category")
    )
    data["week_of_year"] = data.index.isocalendar().week.astype(str).astype("category")
    logging.info(f"data:{data.head()}")
    logging.info(f"data:{data.describe()}")

    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    # data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    data.sample(10, random_state=521)

    max_prediction_length = 6
    max_encoder_length = 24
    val_idx = max(int(len(data) * 0.7), len(data) - 2048 * 16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    training_cutoff = val_idx
    train_data = data[:val_idx]
    logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="close",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length
        // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        static_reals=[],
        allow_missing_timesteps=True,
        time_varying_known_categoricals=[
            "month",
            "hour_of_day",
            "day_of_week",
            "week_of_month",
        ],
        # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "close",
            "volume",
            "log_volume",
            "avg_volume_by_ticker",
        ],
        target_normalizer=GroupNormalizer(
            groups=["ticker"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    MAE()(baseline_predictions.output, baseline_predictions.y)

    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator="cpu",
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=8,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        loss=QuantileLoss(),
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate
    from lightning.pytorch.tuner import Tuner

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
