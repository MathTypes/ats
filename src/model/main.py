import logging
import numpy as np
from pathlib import Path
from pipelines import (
    AttentionLSTMPipeline,
    VanillaLSTMPipeline,
    EmbeddingLSTMPipeline,
    AttentionEmbeddingLSTMPipeline
)
from utils import count_parameters
from torchfitter.io import save_pickle
from torchfitter.utils.convenience import get_logger
from util import logging_utils

RESULTS_PATH = Path("results")


#logger = get_logger(name="Experiments")
#level = logger.level
#logging.basicConfig(level=level)
logging_utils.init_logging()

if __name__ == "__main__":
    datasets = ["stock_returns"]
    pipelines = [
        #VanillaLSTMPipeline,
        #AttentionLSTMPipeline,
        #EmbeddingLSTMPipeline,
        AttentionEmbeddingLSTMPipeline
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

            #y_pred = pipe.preds
            #y_test = pipe.tests
            #history = pipe.history

            #pipe_folder = folder / f"{pip_name}"
            #pipe_folder.mkdir(exist_ok=True)

            #np.save(file=pipe_folder / "y_pred", arr=y_pred)
            #np.save(file=pipe_folder / "y_test", arr=y_test)

            #save_pickle(obj=history, path=pipe_folder / "history.pkl")
