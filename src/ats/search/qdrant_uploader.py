# Import client library
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from vss.metrics.indexer import create_collection, upload_indexes
from vss.metrics.consts import MetricCollections

client = QdrantClient(host="localhost", port=6333)

client.get_collections()

create_collection(
    collection_name=MetricCollections.FUTURES,
    vector_size=1024,
    distance="Cosine"
)

upload_indexes(
    collection_name=MetricCollections.FUTURES,
    meta_file=Path("src/vss/data/qdrant_storage") / f"meta_{MetricCollections.FUTURES.value}.csv",
    dataset_dir=Path("src/vss/data/metric_datasets") / MetricCollections.FUTURES.value,
    qdrant_batch=256,
    meta_filter=None,
)

client.get_collections()

my_collection_info = client.http.collections_api.get_collection(MetricCollections.FUTURES.value)
print(my_collection_info.dict())
