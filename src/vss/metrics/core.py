
from __future__ import annotations

from typing import Union
from dataclasses import dataclass
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
import torch
import torchvision
from loguru import logger
import numpy as np
from PIL import Image
from qdrant_client.grpc import ScoredPoint
from torchvision.transforms.transforms import Compose
from tqdm.auto import tqdm

from ats.app.env_mgr import EnvMgr
from ats.market_data.market_data_mgr import MarketDataMgr
from ats.util import tensor_utils

from vss.common import env_handler, qdrant_client
from vss.common.utils import singleton
from vss.metrics.consts import (
    DEVICE,
    INFER_TRANSFORM,
    METRIC_COLLECTION_NAMES,
    RESIZE_TRANSFORM,
    SEARCH_RESPONSE_LIMIT,
    MetricCollections,
)
from vss.metrics.nets import MODEL_TYPE, get_full_pretrained_model


class InvalidCollectionName(Exception):
    """Exception raised when name of collection name is invalid"""

def decode_file_path(key):
    data = key.split("_")
    return data

@dataclass
class MetricModel:
    model: MODEL_TYPE
    transformation: Compose  # TODO: check if we can define it inside model meta.json files (serialize Compose)

    def __post_init__(self) -> None:
        self.model.eval()


def init_all_metric_models(md_mgr) -> dict[str, MetricModel]:
    """Load all metrics models into memory and return in form of dict"""
    logger.info(f"Loading metric models: {METRIC_COLLECTION_NAMES}")
    return {
        collection_name.value: MetricModel(
            model=get_full_pretrained_model(
                md_mgr,
                collection_name=collection_name, data_parallel=False
            ),
            transformation=INFER_TRANSFORM,
        )
        #for collection_name in tqdm(MetricCollections)
        for collection_name in tqdm(MetricCollections)
    }


#@singleton
class MetricClient:
    """Main client written as a simple bridge between metric search and api"""

    market_map = dict()
    interp_map = dict()
    
    def __init__(self, cfg, device_name: str = DEVICE) -> None:
        self.cfg = cfg
        self.device = device_name
        run_id = None
        self.env_mgr = EnvMgr(cfg, run_id)
        logging.info(f"cfg.job.dataset_transform:{cfg.job.dataset_transform}")
        self.md_mgr = MarketDataMgr(env_mgr=self.env_mgr, transform=cfg.job.dataset_transform)
        self.data_module = self.md_mgr.data_module
        self.models = init_all_metric_models(self.md_mgr)
        local_collection_dir = Path(f"{cfg.job.search_metric_data_dir}")
        logging.error(f"local_collection_dir:{local_collection_dir}")
        for file in local_collection_dir.iterdir():
            file = file.as_posix()
            time_idx = file.split("/")[-1].split("_")[0]
            logging.error(f"checking {file}, time_idx:{time_idx}")
            if ".market." in file:
                self.market_map[time_idx] = file
            else:
                self.interp_map[time_idx] = file

    def _single_img_infer(self, model: MetricModel, ticker: str, decoder_time_idx: int) -> list[float]:
        """Perform single inference of image with proper model and return embeddings"""
        model = model.model
        logging.error(f"decoder_time_idx:{decoder_time_idx}, ticker:{ticker}")
        eval_dataset = self.md_mgr.data_module().validation
        filtered_dataset = eval_dataset.filter(
            lambda x: x.time_idx_first_prediction==decoder_time_idx
        )
        with torch.no_grad():
            val_x, val_y = next(iter(filtered_dataset.to_dataloader()))
            logging.error(f"val_x:{val_x}")
            logging.error(f"val_y:{val_y}")
            if torch.cuda.is_available():
                output = model(tensor_utils.to_cuda(val_x))
            else:
                output = model(val_x)
            logging.error(f"output:{output}")
            embedding = output["embedding"]
            embedding = torch.reshape(embedding, (embedding.size(0), -1))
            embedding = embedding[0]
            logging.error(f"embedding:{embedding.shape}")
        return embedding

    def search_by_ticker_time_idx(
        self,
        ticker: str,
        timestamp: float,
        collection_name: Union[str, MetricCollections],
        limit: int = SEARCH_RESPONSE_LIMIT,
    ) -> list[ScoredPoint]:
        """Search for most similar images (vectors) using qdrant engine"""
        logging.error(f"collection_name:{collection_name}")
        if isinstance(collection_name, MetricCollections):
            collection_name = collection_name.value
        if collection_name not in METRIC_COLLECTION_NAMES:
            raise InvalidCollectionName(collection_name)

        model = self.models[collection_name]
        embedding = self._single_img_infer(model, ticker, decoder_time_idx).detach().cpu().numpy()
        try:
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=embedding,
                query_filter=None,  # TODO: add filtering feature
                limit=limit,
                score_threshold=0.5,
                append_payload=True
            )
            #logging.error(f"search_result:{search_result}")
        except Exception as e:
            logging.error(f"can not search collection:{collection_name}, embedding:{embedding}, limit:{limit}, e:{e}")
        return search_result

    def search(
        self,
        img_path: str,
        collection_name: Union[str, MetricCollections],
        limit: int = SEARCH_RESPONSE_LIMIT,
    ) -> list[ScoredPoint]:
        """Search for most similar images (vectors) using qdrant engine"""
        decoder_time_idx = int(decode_file_path(img_path)[0])
        return self.search_by_ticker_time_idx(ticker, decoder_time_idx, collection_name, limit)


@dataclass
class BestChoiceImagesDataset:
    similars: list[Image.Image]
    results: list[ScoredPoint]
    
    @classmethod
    def get_best_choice_for_uploaded_image(
        cls,
        client: MetricClient,
        key: str,
        anchor: Image.Image,
        collection_name: MetricCollections,
        benchmark: int,
        k: int = 25,
    ) -> BestChoiceImagesDataset:
        """
        Search for similar images of random image from given collection.
        Returns tuple of images [anchor_image, grid image of k most similar images (the biggest cosine similarity)]
        """
        logging.info(f"key:{key}")
        results_all = client.search(collection_name=collection_name, img_path=key, limit=k)
        results = [r for r in results_all if round(r.score, 4) >= benchmark / 100]

        similars = None
        if len(results) > 0:
            imgs = [
                RESIZE_TRANSFORM(img)
                for img in env_handler.get_best_score_imgs(results=results, client=client)
            ]
            to_image = torchvision.transforms.ToPILImage()
            similars = [to_image(img) for img in imgs]

        return cls(
            similars=similars,
            results=results,
        )
