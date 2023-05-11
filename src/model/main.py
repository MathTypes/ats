import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pipelines import (
    TFTPipeline
)
from utils import count_parameters
from torchfitter.io import save_pickle
from torchfitter.utils.convenience import get_logger
from util import logging_utils

RESULTS_PATH = Path("results")

logging_utils.init_logging()

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    datasets = ["stock_returns"]
    pipelines = [
        TFTPipeline
    ]

    for key in datasets:
        folder = RESULTS_PATH / f"{key}"
        folder.mkdir(exist_ok=True)

        for _pipe in pipelines:
            pipe = _pipe(dataset=key)
            pip_name = _pipe.__name__
            logging.info(f"TRAINING: {pip_name}")
            pipe.create_model()
            logging.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")
            pipe.train_model()
