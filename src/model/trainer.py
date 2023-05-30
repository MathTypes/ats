import logging
import os
import warnings
import datetime
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting.data.examples import get_stallion_data
from datasets import generate_stock_returns
from util import logging_utils

from math import ceil

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom/7.0))

if __name__ == "__main__":
    logging_utils.init_logging()

    raw_data = pd.read_parquet("data/token/FUT/30min/ES", engine='fastparquet')
    data = raw_data[["ClosePct", "VolumePct"]]
    data = data.rename(columns={"ClosePct":"close", "VolumePct":"volume"})    
    data["Time"] = data.index
    data["ticker"] = "ES"
    #data["volume"]=data["Volume"]
    #data["close"]=data["Close"]
    data["Time"] = data["Time"].apply(lambda x:x.timestamp()).astype(np.float32)
    logging.info(f"data:{data.head()}")

    data["date"] = data.index
    # add time index
    data.insert(0, 'time_idx', range(0, len(data)))
    #data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    #data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["date_str"] = data.date.apply(lambda x: x.strftime("%Y%U"))
    data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
    data["series"]=data.apply(lambda x: x.ticker + "_"  + x.date_str, axis=1)    
    #data["log_volume"] = np.log(data.volume + 1e-8)
    #data["avg_volume_by_ticker"] = data.groupby(["time_idx", "ticker"], observed=True).volume.transform("mean")
    data["hour_of_day"] = data["date"].apply(lambda x:x.hour).astype(str).astype("category")
    data["day_of_week"] = data.index.dayofweek.astype(str).astype("category")
    data["day_of_month"] = data.index.day.astype(str).astype("category")
    data["week_of_month"] = data["date"].apply(week_of_month).astype(str).astype("category")
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
    #data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    #data.sample(10, random_state=521)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    max_prediction_length = 6
    max_encoder_length = 24
    val_idx = max(int(len(data) * 0.7), len(data) - 2048*16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    training_cutoff = val_idx
    train_data = data[:val_idx]
    logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="close",
        group_ids=["series"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        static_reals=[],
        allow_missing_timesteps=True,
        time_varying_known_categoricals=["month", "hour_of_day", "day_of_week", "week_of_month"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "close",
            "volume",
            #"log_volume",
            #"avg_volume_by_ticker",
        ],
        #target_normalizer=GroupNormalizer(
        #    groups=["se"], transformation="softplus"
        #),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

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
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    tft.hparams.learning_rate = res.suggestion()

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.3,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    logging.info(f'best_model_path:{best_model_path}')

    # calcualte mean absolute error on validation set
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    metrics = MAE()(predictions.output, predictions.y)
    logging.info(f"metrics:{metrics}")

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    fig, axs = plt.subplots(8)
    fig.suptitle('Vertically stacked subplots')
    for idx in range(8):  # plot 10 examples
        time_idx_val = validation.x_to_index(raw_predictions.x)["time_idx"][idx]
        time = data[data.time_idx==time_idx_val]["Time"][0]
        time = datetime.datetime.fromtimestamp(time)
        best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True, ax=axs[idx])
        axs[idx].set_title(str(time))
    plt.show()

    exit(0)
    # calcualte metric by which to display
    predictions = best_tft.predict(val_dataloader, return_y=True)
    mean_losses = SMAPE(reduction="none")(predictions.output, predictions.y).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    for idx in range(10):  # plot 10 examples
        best_tft.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=indices[idx],
            add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles),
        )

    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

    best_tft.predict(
        training.filter(lambda x: (x.time_idx_first_prediction == 15)),
        mode="quantiles",
    )

    raw_prediction = best_tft.predict(
        training.filter(lambda x: (x.time_idx_first_prediction == 15)),
        mode="raw",
        return_x=True,
    )
    best_tft.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0)

    # select last 24 months from data (max_encoder_length is 24)
    encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    # select last known data point and create decoder data from it by repeating it and incrementing the month
    # in a real world dataset, we should not just forward fill the covariates but specify them to account
    # for changes in special days and prices (which you absolutely should do but we are too lazy here)
    last_data = data[lambda x: x.time_idx == x.time_idx.max()]
    decoder_data = pd.concat(
        [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
        ignore_index=True,
    )

    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)

    dependency = best_tft.predict_dependency(
        val_dataloader.dataset, "hour_of_day", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
    )

    #plotting median and 25% and 75% percentile
    agg_dependency = dependency.groupby("hour_of_day").normalized_prediction.agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    ax = agg_dependency.plot(y="median")
    ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3)

