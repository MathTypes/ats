import logging
import os
import warnings
import datetime
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, NHiTS, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, MultiLoss
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
    pd.set_option('display.max_columns', None)
    data = pd.read_parquet("data/token/FUT/30min/ES", engine='fastparquet')
    data["Time"] = data.index
    data["ticker"] = "ES"
    data["volume"]=data["VolumePct"]
    data["close"]=data["ClosePct"]
    data["Time"] = data["Time"].apply(lambda x:x.timestamp()).astype(np.float32)
    logging.info(f"data:{data.head()}")

    data["date"] = data.index
    # add time index
    data.insert(0, 'time_idx', range(0, len(data)))
    #data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    #data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
    data["year"] = data.date.dt.year.astype(str).astype("category")  # categories have be strings
    data["series"]=data.apply(lambda x: x.ticker + "_"  + x.year, axis=1)    
    #data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_ticker"] = data.groupby(["time_idx", "ticker"], observed=True).volume.transform("mean")
    data["hour_of_day"] = data["date"].apply(lambda x:x.hour).astype(str).astype("category")
    data["day_of_week"] = data.index.dayofweek.astype(str).astype("category")
    data["day_of_month"] = data.index.day.astype(str).astype("category")
    data["week_of_month"] = data["date"].apply(week_of_month).astype(str).astype("category")
    data["week_of_year"] = data.index.isocalendar().week.astype(str).astype("category")
    logging.info(f"data:{data.head()}")
    #logging.info(f"data:{data.describe()}")
    logging.info(f"data:{data.info()}")

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

    max_encoder_length = 60
    max_prediction_length = 20
    val_idx = max(int(len(data) * 0.7), len(data) - 2048*16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    training_cutoff = val_idx
    train_data = data[:val_idx]
    context_length = max_encoder_length
    prediction_length = max_prediction_length

    #logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target=["close", "volume"],
        group_ids=["series"],
        #min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=context_length,
        #min_prediction_length=1,
        max_prediction_length=prediction_length,
        #static_categoricals=["ticker"],
        #static_reals=[],
        #allow_missing_timesteps=True,        
        time_varying_known_categoricals=["month", "hour_of_day", "day_of_week", "week_of_month"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        #variable_groups={},  # group of categorical variables can be treated as one variable
        #time_varying_known_reals=["time_idx"],
        #time_varying_known_reals=["hour_of_day", "day_of_week", "week_of_month", "month"],
        #time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["close", "volume"],
        categorical_encoders={
            'series': NaNLabelEncoder(add_nan=True).fit(train_data.series),
            'month': NaNLabelEncoder(add_nan=True).fit(train_data.month),
            'hour_of_day': NaNLabelEncoder(add_nan=True).fit(train_data.hour_of_day),
            'day_of_week': NaNLabelEncoder(add_nan=True).fit(train_data.day_of_week),
            'week_of_month': NaNLabelEncoder(add_nan=True).fit(train_data.week_of_month),
        },
        #target_normalizer=GroupNormalizer(
        #    groups=["series"], transformation="softplus"
        #),  # use softplus and normalize by group
        #add_relative_time_idx=True,
        #add_target_scales=True,
        #add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    #baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    #MAE()(baseline_predictions.output, baseline_predictions.y)

    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator=device,
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )

    net = NHiTS.from_dataset(
        training,
        learning_rate=3e-2,
        weight_decay=1e-2,
        #loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
        loss=MultiLoss(metrics=[MQF2DistributionLoss(prediction_length=max_prediction_length),
        MQF2DistributionLoss(prediction_length=max_prediction_length)], weights=[2.0, 1.0]),
        backcast_loss_ratio=0.0,
        hidden_size=64,
        optimizer="AdamW",
    )

    print(f"Number of parameters in network: {net.size()/1e3:.1f}k")
    res = Tuner(trainer).lr_find(
        net.to(device),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    net.hparams.learning_rate = res.suggestion()

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator=device,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )
    # fit network
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path).to(device)
    logging.info(f'best_model_path:{best_model_path}')

    # calcualte mean absolute error on validation set
    predictions = best_model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator=device))
    #MAE()(predictions.output, predictions.y)

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator=device))

    #ticker = validation.x_to_index(raw_predictions.x)["ticker"]
    fig, axs = plt.subplots(8)        
    fig.suptitle('Vertically stacked subplots')
    logging.info(f"x:{raw_predictions.x}")
    logging.info(f"x.encoder_cat.shape:{raw_predictions.x['encoder_cat'].shape}")    
    for idx in range(8):  # plot 10 examples
        time_idx_val = validation.x_to_index(raw_predictions.x)["time_idx"][idx]
        time = data[data.time_idx==time_idx_val]["Time"]
        time = datetime.datetime.fromtimestamp(time)
        #logging.info(f"validation.x_to_index(raw_predictions.x):{validation.x_to_index(raw_predictions.x)}")
        fig1, fig2 = best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=False, ax=axs[idx])
        axs[idx].set_title(str(time))
        print(f"fig:{fig1}, {fig2}")
        #filename = f"/tmp/file_{idx}.png"
        #fig.savefig(filename)
        #img = mpimg.imread(filename)
    plt.show()
        #imgplot = plt.imshow(img)
        #plt.show()
        


    samples = best_model.loss.sample(raw_predictions.output["prediction"][[0]], n_samples=500)[0]

    # plot prediction
    time_idx_val = validation.x_to_index(raw_predictions.x)["time_idx"][0]
    time = data[data["time_idx"]==time_idx_val]["Time"]
    time = datetime.datetime.fromtimestamp(time)
    fig1, fig2 = best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
    st.write(f"Time: {time}")
    st.pyplot(fig1)
    st.pyplot(fig2)
    ax = fig.get_axes()[0]
    # plot first two sampled paths
    ax.plot(samples[:, 0], color="g", label="Sample 1")
    ax.plot(samples[:, 1], color="r", label="Sample 2")
    fig.legend()

    print(f"fig:{fig}")
    filename = "/tmp/file.png"
    fig.savefig(filename)
    img = mpimg.imread(filename)
    #plt.imshow()
    imgplot = plt.imshow(img)
    plt.suptitle(f"Time: {time}")
    plt.show()

    print(f"Var(all samples) = {samples.var():.4f}")
    print(f"Mean(Var(sample)) = {samples.var(0).mean():.4f}")

    plt.hist(samples.sum(0).numpy(), bins=30)
    plt.xlabel("Sum of predictions")
    plt.ylabel("Frequency")

    plt.show()