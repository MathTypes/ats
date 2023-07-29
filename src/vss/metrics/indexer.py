import logging
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from qdrant_client.conversions.common_types import Distance
from qdrant_client.http import models
from tqdm.auto import tqdm

from ats.util import tensor_utils
from vss.common import qdrant_client
from vss.metrics.consts import INFER_TRANSFORM, MetricCollections
from vss.metrics.nets import get_full_pretrained_model
from vss.metrics.utils import DatasetCombined

DISTANCES = Literal["Cosine", "Euclid", "Dot"]


def shoes_filter(meta: list) -> list:
    """Filter out most of the payload keys to prevent json decode error"""
    new_meta = []
    important_keys = {"file", "class", "label"}
    for d in meta:
        new_meta.append({k: d[k] for k in d.keys() & important_keys})
    return new_meta


def delete_collection(
        collection_name: MetricCollections
) -> None:
    qdrant_client.delete_collection(collection_name=collection_name.value)

# TODO: check if we need wrapper func for singleton client
def create_collection(
        cfg,
        collection_name: MetricCollections,
        vector_size: int,
        distance: Union[Distance, DISTANCES],
) -> None:
    """Wrapper function for auto-injecting qdrant client object and creating collection"""
    qdrant_client.recreate_collection(
        collection_name=collection_name.value,
        vectors_config=models.VectorParams(size=vector_size, distance=distance),
    )


# TODO: add support for batch gpu inference to speed up index upload
def upload_indexes(
        md_mgr,
        collection_name: MetricCollections,
        meta_file: Union[Path, str],
        dataset_dir: Union[Path,  str],
        qdrant_batch: int = 256,
        meta_filter: Optional[Callable] = None,
) -> None:
    """Helper function for creating embeddings and uploading them to qdrant"""
    logger.info(f"Loading model: {collection_name.value}")
    config = md_mgr.config
    model = get_full_pretrained_model(
        md_mgr,
        collection_name=collection_name, data_parallel=False)
    model.eval()
    dataset = DatasetCombined.get_dataset(md_mgr, meta_file, dataset_dir)
    embeddings = []
    meta_data = []
    df = dataset.df
    df = df.fillna("")  # JSON does not support np.nan and pd.NaN
    logger.info(
        f"Started indexing {len(df)} vectors for collection {collection_name.value}"
    )
    # Need train=False to avoid shuffle
    train_dataloader = dataset.train_dataset.to_dataloader(train=False, batch_size=64)
    test_dataloader = dataset.test_dataset.to_dataloader(train=False, batch_size=64)
    matched_eval_data = md_mgr.data_module().eval_data
    logging.info(f"matched_eval_data:{matched_eval_data.iloc[:10]}")
    logging.info(f"matched_eval_data:{matched_eval_data.describe()}")
    for val_x, val_y in iter(train_dataloader):
    #for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        #img = INFER_TRANSFORM(Image.open(row["file"]).convert("RGB"))
        with torch.no_grad():
            if torch.cuda.is_available():
                output = model(tensor_utils.to_cuda(val_x))
            else:
                output = model(val_x)
            logging.info(f"output:{output}")
            embedding = output["embedding"]
            embedding = torch.reshape(embedding, (embedding.size(0), -1))
        logging.info(f"embedding:{embedding.shape}")
        embeddings.extend(embedding.cpu().data.numpy())
        for idx in range(embedding.size(0)):
            decoder_time_idx = int(val_x["decoder_time_idx"][idx][0].cpu().detach().numpy())
            #logging.info(f"checking {decoder_time_idx}")
            train_data_row = matched_eval_data[
                matched_eval_data.time_idx == decoder_time_idx
            ].iloc[0]
            logging.info(f"adding {train_data_row}")
            meta_data_row = dict(train_data_row)
            meta_data_row = tensor_utils.np_to_native(meta_data_row)
            meta_data.append(meta_data_row)
    embeddings = np.array(embeddings)

    if meta_filter:
        meta_data = meta_filter(meta_data)

    qdrant_client.upload_collection(
        collection_name=collection_name.value,
        vectors=embeddings,
        payload=meta_data,
        ids=None,
        batch_size=qdrant_batch,
    )
