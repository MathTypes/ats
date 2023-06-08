# Usage
# PYTHONPATH=.. python3 trainer_nhits_with_dp.py --start_date=2009-01-01 --end_date=2009-10-01 --mode=train --checkpoint=checkpoint
import logging
import os
import warnings
import ray
import time
import wandb
import lightning.pytorch as pl
#from import lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from ray.util.dask import enable_dask_on_ray
from ray_lightning import RayStrategy
from ray.data import ActorPoolStrategy
import datetime
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import matplotlib as mpl
import copy
from pathlib import Path
import warnings
import pyarrow.dataset as pds
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
import lightning.pytorch as pl
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
from log_prediction import LogPredictionsCallback, LSTMLogPredictionsCallback
from data_module import LSTMDataModule, TransformerDataModule, TimeSeriesDataModule

from pytorch_forecasting.data.examples import get_stallion_data
from datasets import generate_stock_returns
from util import logging_utils
from util import config_utils
import nhits_tuner
from math import ceil
from util import time_util
import data_util

def get_model(config, data_module):
    device = config['device']
    training = data_module.training
    max_prediction_length = config['max_prediction_length']
    # configure network and trainer
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    net = NHiTS.from_dataset(
        training,
        learning_rate=3e-2,
        weight_decay=1e-2,
        loss = MQF2DistributionLoss(prediction_length=max_prediction_length),
        backcast_loss_ratio=0.0,
        hidden_size=8,
        optimizer="AdamW",
    )
    return net


def get_trainer(config, data_module):
    device = config['device']
    use_gpu = device == "cuda"
    logging.info(f"device:{device}, use_gpu:{use_gpu}")
    #strategy = RayStrategy(num_workers=config['num_workers'],
    #                       num_cpus_per_worker=1, use_gpu=use_gpu)
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    wandb_logger = WandbLogger(project='ATS', log_model=True)
    logger = TensorBoardLogger(config['model_path'])  # logging results to a tensorboard
    log_predictions_callback = LSTMLogPredictionsCallback(wandb_logger, [data_module.X_test, data_module.y_test])
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=device,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback,
                   #log_predictions_callback
        ],
        #strategy=strategy,
        strategy = "auto",
        logger=wandb_logger,
    )
    return trainer


def run_tune(config, net, trainer, data_module):
    study = nhits_tuner.optimize_hyperparameters(
        data_module.train_dataloader, data_module.val_dataloader,
        config['model_path'],
        config['max_epochs'],
        config['n_trials'])
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
    since  = config["start_date"]
    until = config["end_date"]
    train_data = data_util.get_processed_data(config, ticker, "FUT")
    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna()
    train_data = train_data.drop(columns=["time_idx"])
    return train_data


def get_data_module(config):
    start = time.time()
    train_data_vec = []
    for ticker in config["model_tickers"]:
        ticker_train_data = get_input_for_ticker(config, ticker)
        ticker_train_data["new_idx"] = ticker_train_data.apply(lambda x : x.ticker + "_" + str(x.series_idx), axis=1)
        ticker_train_data = ticker_train_data.set_index("new_idx")
        train_data_vec.append(ticker_train_data)
    train_data = pd.concat(train_data_vec)

    train_data.insert(0, 'time_idx', range(0, len(train_data)))
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
    data_module = TimeSeriesDataModule(config, train_data)
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


if __name__ == "__main__":
    logging_utils.init_logging()
    pd.set_option('display.max_columns', None)
    pd.set_option('use_inf_as_na',True) 
    parser = config_utils.get_arg_parser("Trainer")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--ray_url", type=str, default="ray://8.tcp.ngrok.io:10243")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date(),
        required=False,
        help="Set a end date",
    )
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
        'model_tickers': ['ES','NQ','CL','RTY','HG'],
        'raw_dir': '.',
        'num_workers': 8,
        'device' : args.device,
        'workers': args.workers,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'max_encoder_length' : 13*7,
        'max_prediction_length' : 13,
        'min_encoder_length' : 13*7,
        'min_prediction_length' : 13,
        'context_length' : 13*7,
        'prediction_length' : 13,
        'max_epochs' : args.max_epochs,
        'n_trials' : args.n_trials,
        'model_path' : 'checkpoint'}
    data_module = get_data_module(config)
    net = get_model(config, data_module)
    trainer = get_trainer(config, data_module)
    if args.mode == "tune":
        run_tune(config, net, trainer, data_module)
    elif args.mode == "train":
        run_train(config, net, trainer, data_module)
    elif args.mode == "eval":
        run_eval(config, net, data_module)

    time.sleep(10)
    ray.shutdown()
