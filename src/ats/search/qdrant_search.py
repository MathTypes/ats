# Import client library
import logging
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

from ats.util import logging_utils

logging_utils.init_logging()

client = QdrantClient(host="localhost", port=6333)

client.get_collections()

query_vector = np.random.rand(1024)
hits = client.search(
    collection_name=MetricCollections.FUTURES.value,
    query_vector=query_vector,
    query_filter=None,  # Don't use any filters for now, search across all indexed points
    append_payload=True,  # Also return a stored payload for found points
    limit=5  # Return 5 closest points
)

logging.info(f"{hits}")

