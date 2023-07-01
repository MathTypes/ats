# Usage
# PYTHONPATH=.. python3 trainer_nhits_with_dp.py --start_date=2009-01-01 --end_date=2009-10-01 --mode=train --checkpoint=checkpoint
import datetime
import logging
import os
import warnings
import ray
import time
import wandb
import copy
from pathlib import Path
import warnings
import pyarrow.dataset as pds
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
from scipy.signal import argrelmax,argrelmin, argrelextrema, find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from omegaconf import OmegaConf

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, PatchTstTransformer, PatchTstTftTransformer, PatchTstTftSupervisedTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, SharpeLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_lightning.loggers import WandbLogger
from pytorch_forecasting import Baseline, NHiTS, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, MAPCSE, RMSE, SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, MultiLoss

from ray.util.dask import enable_dask_on_ray
from ray_lightning import RayStrategy
from ray.data import ActorPoolStrategy

from data_module import LSTMDataModule, TransformerDataModule, TimeSeriesDataModule
from datasets import generate_stock_returns
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import matplotlib as mpl
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from wandb.keras import WandbMetricsLogger

from torch import nn
import data_util
from log_prediction import LogPredictionsCallback, LSTMLogPredictionsCallback
from math import ceil
import nhits_tuner
from util import logging_utils
from util import config_utils
from util import time_util

def create_loss(loss_name, device, prediction_length=None, hidden_size=None):
    loss = None
    logging.info(f"loss_name:{loss_name}")
    if loss_name == "MASE":
        loss = MASE()
    if loss_name == "SharpeLoss":
        loss = SharpeLoss()
    if loss_name == "SMAPE":
        loss = SMAPE()
    if loss_name == "MAE":
        loss = MAE()
    if loss_name == "RMSE":
        loss = RMSE()
    if loss_name == "MAPE":
        loss = MAPE()
    if loss_name == "MAPCSE":
        loss = MAPCSE()
    if loss_name == "MQF2DistributionLoss":
        loss = MQF2DistributionLoss(prediction_length=prediction_length)
    loss = loss.to(device)
    return loss

def get_logging_metrics(config):
    metrics = []
    for loss_name in config.model.logging_metrics:
        metrics.append(create_loss(loss_name, config.job.device, config.model.prediction_length))
    return nn.ModuleList(metrics)

def get_loss(config, prediction_length=None, hidden_size=None):
    target_size = 1
    is_multi_target = False
    if not prediction_length:
        prediction_length = config.model.prediction_length
    #logging.info(f"prediction_length:{prediction_length}")
    if OmegaConf.is_list(config.model.target):
        target = OmegaConf.to_object(config.model.target)
        target_size = len(target)
        is_multi_target = True
    loss_name = config.model.loss_name
    loss = create_loss(loss_name, config.job.device, prediction_length)
    return loss


def get_model(config, data_module):
    device = config['device']
    training = data_module.training
    max_prediction_length = config['max_prediction_length']
    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    loss = get_loss(config)
    net = NHiTS.from_dataset(
        training,
        weight_decay=1e-2,
        loss = loss,
        #loss = MQF2DistributionLoss(prediction_length=max_prediction_length),
        #loss=MultiLoss(metrics=[MQF2DistributionLoss(prediction_length=max_prediction_length),
                                #MQF2DistributionLoss(prediction_length=max_prediction_length),
                                #MQF2DistributionLoss(prediction_length=max_prediction_length)],
        #                        ],
        #               weights=[1.0
                                #, 0.0, 0.0
        #               ]),
        backcast_loss_ratio=0.0,
        hidden_size=config["hidden_size"],
        prediction_length=config["prediction_length"],
        context_length=config["context_length"],
        learning_rate=config["learning_rate"],
        optimizer="AdamW",
        log_interval=0.25,
        #n_blocks=[1,1,1],
        #downsample_frequencies=[1,23,46],
        #n_layers=2,
        #log_val_interval=10000
    )
    return net

def get_tft_model(config, data_module):
    device = config.job.device
    training = data_module.training
    max_prediction_length = config.model.prediction_length
    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    loss = get_loss(config)
    #logging.error(f"loss:{loss}")
    net = TemporalFusionTransformer.from_dataset(
        training,
        weight_decay=1e-2,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=8,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        loss=loss,
        optimizer="Ranger",
        log_interval=0.25,
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=10,
    )
    return net

def get_patch_tst_model(config, data_module):
    device = config.job.device
    training = data_module.training
    prediction_length = config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    loss = get_loss(config)
    num_patch = (max(context_length, patch_len)-patch_len) // stride + 1
    logging.getLogger().info(f"context_length:{context_length}, patch_len:{patch_len}, stride:{stride}, num_patch:{num_patch}")
    net = PatchTstTransformer.from_dataset(
        training,
        patch_len=config.model.patch_len,
        stride=stride,
        num_patch=num_patch,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        d_model=8,  # most important hyperparameter apart from learning rate
        hidden_size=8,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=1,
        loss=loss,
        attn_dropout=0.1,  # between 0.1 and 0.3 are good values
        #hidden_continuous_size=8,  # set to <= hidden_size
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net

def get_patch_tst_tft_model(config, data_module):
    device = config.job.device
    training = data_module.training
    prediction_length = config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    pl.seed_everything(42)
    d_model = config.model.d_model
    #patch_prediction_length = int(prediction_length/stride)-1
    logging.info(f"prediction_length:{prediction_length}, patch_len:{patch_len}")
    loss = get_loss(config, hidden_size=d_model)
    num_patch = (max(context_length, patch_len)-patch_len) // stride + 1
    logging.info(f"patch_len:{patch_len}, num_patch:{num_patch}, context_length:{context_length}, stride:{stride}")
    net = PatchTstTftTransformer.from_dataset(
        training,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        loss=loss,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        d_model=d_model,  # most important hyperparameter apart from learning rate
        hidden_size=8,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=1,
        attn_dropout=0.1,  # between 0.1 and 0.3 are good values
        #hidden_continuous_size=8,  # set to <= hidden_size
        #loss=QuantileLoss(),
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net

def get_patch_tst_tft_supervised_model(config, data_module):
    device = config.job.device
    training = data_module.training
    prediction_length = config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    d_model = config.model.d_model
    pl.seed_everything(42)
    #patch_prediction_length = int(prediction_length/stride)-1
    logging.info(f"prediction_length:{prediction_length}, patch_len:{patch_len}")
    loss = get_loss(config)
    num_patch = (max(context_length, patch_len)-patch_len) // stride + 1
    prediction_num_patch = (max(prediction_length, patch_len)-patch_len) // stride + 1
    logging.info(f"patch_len:{patch_len}, num_patch:{num_patch}, context_length:{context_length}, stride:{stride}")
    net = PatchTstTftSupervisedTransformer.from_dataset(
        training,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        prediction_num_patch=prediction_num_patch,
        loss=loss,
        logging_metrics=get_logging_metrics(config),
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        d_model=d_model,  # most important hyperparameter apart from learning rate
        hidden_size=8,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=1,
        attn_dropout=0.1,  # between 0.1 and 0.3 are good values
        #hidden_continuous_size=8,  # set to <= hidden_size
        #loss=QuantileLoss(),
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net

def get_trainer(config, data_module):
    device = config['device']
    use_gpu = device == "cuda"
    logging.getLogger().info(f"device:{device}, use_gpu:{use_gpu}")
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    wandb_logger = WandbLogger(project='ATS', log_model=config["log_mode"])
    #logger = TensorBoardLogger(config['model_path'])  # logging results to a tensorboard
    metrics_logger = WandbMetricsLogger(log_freq=10)
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=device,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[#lr_logger,
                   #early_stop_callback,
                   #log_predictions_callback,
                   #metrics_logger,
                   #prediction_logger
        ],
        strategy = "ddp",
        devices=1,
        precision=16,
        logger=wandb_logger,
    )
    return trainer


def run_tune(study_name, config):
    study = nhits_tuner.optimize_hyperparameters(study_name, config)
    print(f"study:{study}")


def get_input_dirs(config):
    base_dir = "data/FUT/30min_rsp/ES"
    input_dirs = []
    for cur_date in time_util.monthlist(config['start_date'], config['end_date']):
        for_date = cur_date[0]
        date_dir = os.path.join(base_dir, for_date.strftime("%Y%m%d"))
        files = os.listdir(date_dir)
        files = [date_dir+'/'+f for f in files if os.path.isfile(date_dir+'/'+f)] #Filtering only the files.
        input_dirs.extend(files)
    return input_dirs


def get_input_for_ticker(config, ticker):
    all_data = data_util.get_processed_data(config, ticker, "FUT")
    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    all_data = all_data.dropna()
    all_data = all_data.drop(columns=["time_idx"])
    return all_data

def add_highs(df, width):
    df_cumsum = df.cumsum()
    high_idx, _ = find_peaks(df_cumsum, width=width)
    high = df_cumsum.iloc[high_idx].to_frame(name="close_cumsum_high")
    df_high = df_cumsum.to_frame(name="close_cumsum").join(high)
    df_high = df_high.bfill()
    return df_high["close_cumsum_high"]

def add_lows(df, width):
    df_cumsum = df.cumsum()
    low_idx, _ = find_peaks(np.negative(df_cumsum), width=width)
    low = df_cumsum.iloc[low_idx].to_frame(name="close_cumsum_low")
    df_low = df_cumsum.to_frame(name="close_cumsum").join(low)
    df_low = df_low.bfill()
    return df_low["close_cumsum_low"]

def get_data_module(config):
    start = time.time()
    train_data_vec = []
    for ticker in config.dataset.model_tickers:
        ticker_train_data = get_input_for_ticker(config, ticker)
        ticker_train_data["new_idx"] = ticker_train_data.apply(lambda x : x.ticker + "_" + str(x.series_idx), axis=1)
        ticker_train_data = ticker_train_data.set_index("new_idx")
        train_data_vec.append(ticker_train_data)
    raw_data = pd.concat(train_data_vec)
    g = raw_data.groupby(["ticker"], observed=True)
    raw_data["close_high_21"] = g['close_back'].transform(add_highs, width=21)
    raw_data["close_low_21"] = g['close_back'].transform(add_lows, width=21)
    raw_data["close_high_51"] = g['close_back'].transform(add_highs, width=51)
    raw_data["close_low_51"] = g['close_back'].transform(add_lows, width=51)
    raw_data["close_high_201"] = g['close_back'].transform(add_highs, width=201)
    raw_data["close_low_201"] = g['close_back'].transform(add_lows, width=201)
    #logging.info(f"raw_data:{raw_data}")
    eval_cut_off = config.job.eval_cut_off
    #logging.info(f"eval_cut_off:{eval_cut_off}")
    train_data = raw_data[raw_data.year<=eval_cut_off]
    eval_data = raw_data[raw_data.year>eval_cut_off]
    train_data = train_data.sort_values(["ticker", "time"])
    eval_data = eval_data.sort_values(["ticker", "time"])
    train_data.insert(0, 'time_idx', range(0, len(train_data)))
    eval_data.insert(0, 'time_idx', range(0, len(eval_data)))
    data_loading_time = time.time() - start
    logging.info(f"Data loading time: {data_loading_time:.2f} seconds")

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
    data_module = TimeSeriesDataModule(config, train_data, eval_data)
    return data_module


def run_train(config, net, trainer, data_module):
    device = config['device']
    # fit network
    trainer.fit(
        net,
        train_dataloaders=data_module.train_dataloader,
        val_dataloaders=data_module.val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path).to(device)
    logging.info(f'best_model_path:{best_model_path}')


def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom/7.0))

