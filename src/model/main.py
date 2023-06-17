import argparse
import datetime
import hydra
import logging
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from pipelines import (
    TFTPipeline,
    AttentionEmbeddingLSTMPipeline,
    TimeSeriesPipeline
)
import pytz
import ray
import wandb
from ray.util.dask import enable_dask_on_ray
from utils import count_parameters
from util import config_utils
from util import logging_utils

RESULTS_PATH = Path("results")

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    wandb.init(project="ats")
    pd.set_option('display.max_columns', None)
    datasets = ["stock_returns"]
    pipelines = [
        #TFTPipeline
        AttentionEmbeddingLSTMPipeline,
        TimeSeriesPipeline
    ]
    #config_utils.set_args(args)
    logging_utils.init_logging()
    logging.info(f"cfg:{cfg}, dir(cfg)")
    ray.init()
    enable_dask_on_ray()
    device = cfg.job.device
    logging.info(f"start_date:{cfg.job.start_date}")
    logging.info(f"end_date:{cfg.job.end_date}")
    context_length = 13*7*5*6
    prediction_length = 13*3
    config = {
        'model_tickers': ['ES','NQ','CL','RTY','HG'],
        'raw_dir': '.',
        'num_workers': 8,
        'device' : cfg.job.device,
        'workers': cfg.job.workers,
        'start_date': cfg.job.start_date,
        'end_date': cfg.job.end_date,
        'max_encoder_length' : context_length,
        'max_prediction_length' : prediction_length,
        'min_encoder_length' : prediction_length,
        'min_prediction_length' : prediction_length,
        'context_length' : context_length,
        'prediction_length' : prediction_length,
        'max_epochs' : cfg.job.max_epochs,
        'n_trials' : cfg.job.n_trials,
        'model_path' : 'checkpoint'}
    wandb.config = config
    pipe = TimeSeriesPipeline(dataset="FUT", device=device, config=config)
    if args.mode == "train":
        pipe.create_model()
        pipe.create_trainer()
        logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
        pipe.train_model()
    elif args.mode == "tune":
        pipe.tune_model(config, args.study_name)
    elif args.mode == "eval":
        pipe.create_model()
        config["checkpoint_path"] = args.checkpoint
        pipe.create_trainer()
        pipe.eval_model(config)
    ray.shutdown()

    
if __name__ == "__main__":
  my_app()
