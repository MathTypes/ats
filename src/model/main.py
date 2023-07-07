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
    TimeSeriesPipeline,
    TemporalFusionTransformerPipeline,
    PatchTstTransformerPipeline,
    PatchTstTftPipeline,
    PatchTftSupervisedPipeline
)
import pytz
import ray
import torch
import wandb
from ray.util.dask import enable_dask_on_ray
from utils import count_parameters
from util import config_utils
from util import logging_utils
import sys

RESULTS_PATH = Path("results")

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    wandb.init(project="ats", name=f"{cfg.model.name}-{cfg.job.tag}", config=cfg)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    datasets = ["stock_returns"]
    pipelines = {
        "attention":AttentionEmbeddingLSTMPipeline,
        "tft": TemporalFusionTransformerPipeline,
        "patch_tst": PatchTstTransformerPipeline,
        "patch_tst_tft": PatchTstTftPipeline,
        "patch_tft_supervised": PatchTftSupervisedPipeline,
        "nhits": TimeSeriesPipeline
    }
    logging_utils.init_logging()
    logging.info(f"cfg:{cfg}, dir(cfg)")
    ray.init()
    enable_dask_on_ray()
    prediction_length = cfg['model']['prediction_length']
    context_length = cfg['model']['context_length']
    wandb.config = cfg
    pipe = pipelines[cfg.model.name](dataset="FUT", config=cfg) 
    if cfg.job.mode == "train":
        pipe.create_trainer()
        pipe.create_model(cfg.job.checkpoint)
        logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
        if cfg.job.tune_learning_rate:
            pipe.set_learning_rate()
        pipe.train_model()
        #pipe.test_model()
    elif cfg.job.mode == "tune":
        config['n_trials'] = cfg.job.n_trials
        pipe.tune_model(config, cfg.job.study_name)
    elif cfg.job.mode == "eval":
        pipe.create_model(cfg.job.checkpoint)
        pipe.eval_model()
    elif cfg.job.mode == "test":
        pipe.create_model(cfg.job.checkpoint)
        pipe.test_model()

    ray.shutdown()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    my_app()
