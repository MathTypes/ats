import kaleido
import datetime
import logging
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
import pandas as pd
import ray
import torch
import wandb
from ray.util.dask import enable_dask_on_ray

from ats.app.pipelines import (
    PatchTstTransformerPipeline,
    PatchTstTftPipeline,
    PatchTftSupervisedPipeline,
    TimeSeriesPipeline,
)
from ats.model.utils import count_parameters
from ats.util import logging_utils

RESULTS_PATH = Path("results")

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:    
    logging.info(f"Current working directory : {os.getcwd()}")
    logging.info(f"Orig working directory    : {get_original_cwd()}")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(
        project="ats", name=f"{cfg.model.name}-{cfg.job.tag}-{run_id}", config=cfg
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:,.4f}".format
    pipelines = {
        "nhits": TimeSeriesPipeline,
        "patch_tst": PatchTstTransformerPipeline,
        "patch_tst_tft": PatchTstTftPipeline,
        "patch_tft_supervised": PatchTftSupervisedPipeline,
    }
    logging_utils.init_logging()
    logging.info(f"cfg:{cfg}, dir(cfg)")
    ray.init(object_store_memory=80*1024*1024*1024,
             storage=f"{cfg.dataset.base_dir}/cache",
             configure_logging=True,
             logging_level=logging.ERROR)
    enable_dask_on_ray()
    cfg["model"]["prediction_length"]
    cfg["model"]["context_length"]
    wandb.config = cfg
    pipe = pipelines[cfg.model.name](dataset="FUT", config=cfg, run_id=run_id)
    checkpoint = cfg.job.checkpoint
    if cfg.job.mode == "train":
        pipe.create_trainer()
        pipe.create_model(checkpoint)
        logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
        if cfg.job.tune_learning_rate:
            pipe.set_learning_rate()
        pipe.train_model()
        # pipe.test_model()
    elif cfg.job.mode == "tune":
        pipe.tune_model(run_id)
    elif cfg.job.mode == "eval":
        pipe.create_model(checkpoint)
        pipe.eval_model()
    elif cfg.job.mode == "build_search":
        pipe.create_model(checkpoint)
        pipe.build_search()
    elif cfg.job.mode == "search":
        pipe.create_model(checkpoint)
        pipe.search_examples()
    elif cfg.job.mode == "test":
        # train model until test start
        pipe.create_trainer()
        pipe.create_model(checkpoint)
        if cfg.job.tune_learning_rate:
            pipe.set_learning_rate()
        if cfg.job.retrain_model_before_test_start:
            pipe.train_model()
        pipe.test_model()

    ray.shutdown()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    my_app()
