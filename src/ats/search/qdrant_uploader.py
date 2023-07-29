# Import client library
import logging
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import pandas as pd
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from qdrant_client.http import models

from ats.app.env_mgr import EnvMgr
from ats.market_data.market_data_mgr import MarketDataMgr
from vss.metrics.indexer import create_collection, upload_indexes
from vss.metrics.consts import MetricCollections

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    env_mgr = EnvMgr(cfg, run_id=None)
    md_mgr = MarketDataMgr(env_mgr)
    client = QdrantClient(host="localhost", port=6333)

    client.get_collections()

    #delete_collection(collection_name=MetricCollections.FUTURES)
    create_collection(
        cfg,
        collection_name=MetricCollections.FUTURES,
        vector_size=288,
        distance="Cosine"
    )

    logging.error(f"uploading from {cfg.job.search_metric_data_dir}")
    upload_indexes(
        md_mgr,
        collection_name=MetricCollections.FUTURES,
        meta_file=cfg.job.search_metric_meta,
        dataset_dir=cfg.job.search_metric_data_dir,
        qdrant_batch=256,
        meta_filter=None,
    )

    client.get_collections()

    my_collection_info = client.http.collections_api.get_collection(MetricCollections.FUTURES.value)
    print(my_collection_info.dict())

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    my_app()
