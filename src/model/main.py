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
    logging.info(f"cfg:{cfg}, dir(cfg)")
    ray.init()
    enable_dask_on_ray()
    device = cfg.job.device
    logging.info(f"start_date:{cfg.job.start_date}")
    logging.info(f"end_date:{cfg.job.end_date}")
    #context_length = 13*7*5*6
    #prediction_length = 13*3
    prediction_length = cfg['model']['prediction_length']
    context_length = cfg['model']['context_length']
    config = {
        'model_tickers': ['ES','NQ','CL','RTY','HG'],
        'raw_dir': '.',
        'base_dir':cfg['dataset']['base_dir'],
        'hidden_size':cfg['model']['hidden_size'],
        'learning_rate':cfg['optimizer']['lr'],
        'num_workers': 8,
        'log_mode': cfg['job']['log_mode'],
        'loss_name': cfg['model']['loss_name'],
        'device' : cfg.job.device,
        'workers': cfg.job.workers,
        'start_date': datetime.datetime.strptime(cfg.job.start_date, "%Y-%m-%d"),
        'end_date': datetime.datetime.strptime(cfg.job.end_date, "%Y-%m-%d"),
        'max_encoder_length' : context_length,
        'max_prediction_length' : prediction_length,
        'min_encoder_length' : prediction_length,
        'min_prediction_length' : prediction_length,
        'context_length' : context_length,
        'prediction_length' : prediction_length,
        'max_epochs' : cfg.job.max_epochs,
        'model_path' : 'checkpoint'}
    wandb.config = config
    pipe = TimeSeriesPipeline(dataset="FUT", device=device, config=config)
    if cfg.job.mode == "train":
        pipe.create_model()
        pipe.create_trainer()
        logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
        pipe.train_model()
    elif cfg.job.mode == "tune":
        config['n_trials'] = cfg.job.n_trials
        pipe.tune_model(config, cfg.job.study_name)
    elif cfg.job.mode == "eval":
        pipe.create_model()
        config["checkpoint_path"] = cfg.job.checkpoint
        pipe.create_trainer()
        pipe.eval_model(config)
    ray.shutdown()

    
if __name__ == "__main__":
  logging_utils.init_logging()
  my_app()
