import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pipelines import (
    TFTPipeline,
    AttentionEmbeddingLSTMPipeline
)
from utils import count_parameters
from torchfitter.io import save_pickle
from torchfitter.utils.convenience import get_logger
from util import config_utils
from util import logging_utils

RESULTS_PATH = Path("results")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    datasets = ["stock_returns"]
    pipelines = [
        #TFTPipeline
        AttentionEmbeddingLSTMPipeline
    ]
    parser = config_utils.get_arg_parser("Preprocess tweet")
    parser.add_argument("device", type=str)
    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    for key in datasets:
        folder = RESULTS_PATH / f"{key}"
        folder.mkdir(exist_ok=True)

        for _pipe in pipelines:
            pipe = _pipe(dataset=key)
            pip_name = _pipe.__name__
            logging.info(f"TRAINING: {pip_name}")
            pipe.create_model(args.device)
            logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
            pipe.train_model()
