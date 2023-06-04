import logging
import os
import warnings
import ray
import time
from ray.util.dask import enable_dask_on_ray
from ray_lightning import RayStrategy
import datetime
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import matplotlib as mpl
import copy
from pathlib import Path
import warnings
import pyarrow.dataset as pds
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
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
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting.data.examples import get_stallion_data
from datasets import generate_stock_returns
from util import logging_utils
from util import config_utils
import nhits_tuner
from math import ceil

def get_model(config, training):
    device = config['device']
    max_prediction_length = config['max_prediction_length']
    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    net = NHiTS.from_dataset(
        training,
        learning_rate=3e-2,
        weight_decay=1e-2,
        #loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
        #loss=MultiLoss(metrics=[MQF2DistributionLoss(prediction_length=max_prediction_length),
        #MQF2DistributionLoss(prediction_length=max_prediction_length)], weights=[2.0, 1.0]),
        loss = MQF2DistributionLoss(prediction_length=max_prediction_length),
        backcast_loss_ratio=0.0,
        hidden_size=8,
        optimizer="AdamW",
    )
    return net

def get_trainer(config):
    device = config['device']
    use_gpu = device == "cuda"
    logging.info(f"device:{device}, use_gpu:{use_gpu}")
    strategy = RayStrategy(num_workers=config['num_workers'],
                           num_cpus_per_worker=1, use_gpu=use_gpu)
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(config['model_path'])  # logging results to a tensorboard
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=device,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        strategy=strategy,
        logger=logger,
    )
    return trainer

def run_tune(config, net, trainer, train_dataloader, val_dataloader):
    study = nhits_tuner.optimize_hyperparameters(
        train_dataloader, val_dataloader,
        config['model_path'],
        config['max_epochs'],
        config['n_trials'])
    print(f"study:{study}")
    #device = config['device']
    #print(f"Number of parameters in network: {net.size()/1e3:.1f}k")
    #res = Tuner(trainer).lr_find(
    #    net.to(device),
    #    train_dataloaders=train_dataloader,
    #    val_dataloaders=val_dataloader,
    #    min_lr=1e-5,
    #    max_lr=1e0,
    #    early_stop_threshold=100,
    #)
    #print(f"suggested learning rate: {res.suggestion()}")
    #fig = res.plot(show=True, suggest=True)
    #fig.show()
    #net.hparams.learning_rate = res.suggestion()

def get_input_data(config):
    start = time.time()
    filter_expr = (
        (pds.field("hour_of_day").isin([3]))
    )
    ds = ray.data.read_parquet("data/FUT/30min_rolled_sampled/ES", parallelism=10, filter=filter_expr)
    data_loading_time = time.time() - start
    logging.info(f"Data loading time: {data_loading_time:.2f} seconds")
    #raw_data = pd.read_parquet("data/FUT/30min_rolled_sampled/ES", engine='fastparquet')
    #data = raw_data[["Close", "Volume"]]
    #data = raw_data
    #data = data.rename(columns={"ClosePct":"close", "VolumePct":"volume"})
    #data["volume"] = data["volume"].ewm(span=60, min_periods=60).std().fillna(method="bfill")
    #data["Time"] = data["time"]
    #data["ticker"] = "ES"
    #data["volume"]=data["VolumePct"]
    #data["Time"] = data["Time"].apply(lambda x:x.timestamp()).astype(np.float32)
    #logging.info(f"data:{data.head()}")
    #data = data.dropna()
    #data["date"] = data.est_time
    #data["time_idx"] = data.index
    # add time index
    #data.insert(0, 'time_idx', range(0, len(data)))
    #data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    #data["time_idx"] -= data["time_idx"].min()

    # add additional features
    #data["date_str"] = data.date.apply(lambda x: x.strftime("%Y%U"))
    #data["series"]=data.apply(lambda x: x.ticker + "_"  + x.date_str, axis=1)    
    #data["series"] = data["id"]
    #data["log_volume"] = np.log(data.volume + 1e-8)
    #data["avg_volume_by_ticker"] = data.groupby(["time_idx", "ticker"], observed=True).volume.transform("mean")
    #data["hour_of_day"] = data["date"].apply(lambda x:x.hour).astype(str).astype("category")
    #logging.info(f"data:{data.head()}")
    #logging.info(f"data:{data.describe()}")
    #logging.info(f"data:{data.info()}")

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
    train_data, test_data = ds.train_test_split(test_size=0.25)
    #data_len = data.count()
    #val_idx = max(int(len(data) * 0.7), len(data) - 2048*16)
    #tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    #training_cutoff = val_idx
    #train_data = data[:val_idx]

    #logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    train_data = train_data.to_dask().compute()
    train_data["time_idx"] = train_data.index
    train_data.index = train_data.apply(lambda x: x.id + "_" + str(x.time_idx), axis=1)
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="Close",
        group_ids=["id"],
        #min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=config['context_length'],
        #min_prediction_length=1,
        max_prediction_length=config['prediction_length'],
        #static_categoricals=["ticker"],
        #static_reals=[],
        allow_missing_timesteps=True,        
        #time_varying_known_categoricals=["month", "hour_of_day", "day_of_week", "week_of_month"],
        #time_varying_known_categoricals=["hour_of_day"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        #variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "hour_of_day"],
        #time_varying_known_reals=[],
        #time_varying_known_reals=["hour_of_day", "day_of_week", "week_of_month", "month"],
        #time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["Close", "Volume"],
        categorical_encoders={
            #'series': NaNLabelEncoder(add_nan=True).fit(train_data.series),
            #'month': NaNLabelEncoder(add_nan=True).fit(train_data.month),
            #'hour_of_day': NaNLabelEncoder(add_nan=True).fit(train_data.hour_of_day),
            #'day_of_week': NaNLabelEncoder(add_nan=True).fit(train_data.day_of_week),
            #'week_of_month': NaNLabelEncoder(add_nan=True).fit(train_data.week_of_month),
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
    test_data = test_data.to_dask().compute()
    test_data["time_idx"] = test_data.index
    test_data.index = test_data.apply(lambda x: x.id + "_" + x.time_idx)
    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=False)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=4, pin_memory=True, drop_last=False)
    return training, train_dataloader, val_dataloader, data


def run_train(config, net, trainer, train_dataloader, val_dataloader, data):
    device = config['device']
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
    #predictions = best_model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator=device))
    #metrics = MAE()(predictions.output, predictions.y)
    #logging.info(f"metrics:{metrics}")

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator=device))

    #ticker = validation.x_to_index(raw_predictions.x)["ticker"]
    #fig, axs = plt.subplots(4, 2)
    fig, axs = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    #logging.info(f"x:{raw_predictions.x}")
    logging.info(f"x.encoder_cat.shape:{raw_predictions.x['encoder_cat'].shape}")
    prediction_kwargs = {"use_metric":False}
    quantiles_kwargs = {"use_metric":False}
    #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 
    for idx in range(4):  # plot 10 examples
        time_idx_val = val_dataloader.dataset.x_to_index(raw_predictions.x)["time_idx"][idx]
        time = data[data.time_idx==time_idx_val]["Time"][0]
        time = datetime.datetime.fromtimestamp(time.astype(int))
        #logging.info(f"validation.x_to_index(raw_predictions.x):{validation.x_to_index(raw_predictions.x)}")
        figs = best_model.plot_prediction(
            raw_predictions.x, raw_predictions.output, idx=idx,
            add_loss_to_title=False, ax=axs[idx],
            show_future_observed=True,
            quantiles_kwargs=quantiles_kwargs,
            prediction_kwargs=prediction_kwargs)
        #figs = best_model.plot_interpretation(raw_predictions.x, raw_predictions.output, idx=idx, ax=[axs[idx, 0], axs[idx,1]])
        #axs[idx, 0].set_title(str(time))
        axs[idx].set_title(str(time))
        #axs[idx, 1].set_title(str(time))
        #print(f"fig:{fig1}, {fig2}")
        #filename = f"/tmp/file_{idx}.png"
        #fig.savefig(filename)
        #img = mpimg.imread(filename)
    plt.show()
        #imgplot = plt.imshow(img)
        #plt.show()

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
    parser = config_utils.get_arg_parser("Trainer")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--ray_url", type=str, default="ray://8.tcp.ngrok.io:10243")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--lr", type=float, default=4.4668359215096314e-05)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)
    logging_utils.init_logging()
    args = parser.parse_args()
    logging.info(f"init from {args.ray_url}")
    #ray.init(args.ray_url)
    ray.init()
    enable_dask_on_ray()
    device = args.device
    config = {
        'device' : args.device,
        'workers': args.workers,
        'max_encoder_length' : 13*14,
        'max_prediction_length' : 13*3,
        'context_length' : 13*14,
        'prediction_length' : 13*3,
        'max_epochs' : args.max_epochs,
        'n_trials' : args.n_trials,
        'model_path' : 'checkpoint'}
    training, train_dataloader, val_dataloader, data = get_input_data(config)
    net = get_model(config, training)
    trainer = get_trainer(config)
    if args.mode == "tune":
        run_tune(config, net, trainer, train_dataloader, val_dataloader)
    elif args.mode == "train":
        run_train(config, net, trainer, train_dataloader, val_dataloader, data)
    elif args.mode == "eval":
        run_eval(config, net, val_dataloader)

