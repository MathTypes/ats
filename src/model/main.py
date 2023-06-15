import argparse
import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pipelines import (
    TFTPipeline,
    AttentionEmbeddingLSTMPipeline,
    TimeSeriesPipeline
)
import pytz
import ray
from ray.util.dask import enable_dask_on_ray
from utils import count_parameters
from util import config_utils
from util import logging_utils

RESULTS_PATH = Path("results")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    datasets = ["stock_returns"]
    pipelines = [
        #TFTPipeline
        AttentionEmbeddingLSTMPipeline,
        TimeSeriesPipeline
    ]
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--ray_url", type=str, default="ray://8.tcp.ngrok.io:10243")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--start_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=pytz.UTC).date(),
        required=False,
        help="Set a start date",
    )
    parser.add_argument(
        "--end_date",
        type=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=pytz.UTC).date(),
        required=False,
        help="Set a end date",
    )
    parser.add_argument("--lr", type=float, default=4.4668359215096314e-05)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()
    ray.init()
    enable_dask_on_ray()
    device = args.device
    logging.info(f"start_date:{args.start_date}")
    logging.info(f"end_date:{args.end_date}")
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
    
    pipe = TimeSeriesPipeline(dataset="FUT", device=args.device, config=config)
    pipe.create_model()
    pipe.create_trainer()
    logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
    if args.mode == "train":
        pipe.train_model()
    elif args.mode == "tune":
        pipe.tune_model()
    ray.shutdown()

    
