import json
import logging
import random
from pathlib import Path, PurePath
import traceback
from PIL import Image
from qdrant_client.grpc import ScoredPoint

from vss.common.handler_env import EnvFunctionHandler
from vss.common.utils import singleton
from vss.metrics.consts import MetricCollections


    
@singleton
class LocalFunctionHandler(EnvFunctionHandler):
    """
    Managing class for local environment methods.
    """

    def get_best_score_imgs(self, results: list, client) -> list:
        """
        Handler for returning images with the highest similarity scores from local storage.
        Additionally, filenames are returned as future captions in front-end module.
        """
        #logging.error(f"results:{results}")
        object_list = []
        for r in results:
            time_idx = r.payload["time_idx"]
            logging.error(f"time_idx:{time_idx}")
            file = client.interp_map[str(time_idx)]
            logging.error(f"time_idx:{time_idx}, file:{file}")
            object_list.append(file)
            logging.info(f"r before:{r}")
            r.payload["file"] = file
            logging.info(f"r after:{r}")
        logging.error(f"object_list:{object_list}")
        return [Image.open(obj.replace("src/vss/","")) for obj in object_list]

    def get_random_images_from_collection(
        self, collection_name: MetricCollections, k: int
    ) -> tuple:
        """
        Pulls a random set of images from a selected collection in local storage.
        Used for image input suggestion in front-end component.
        Additionally, filenames are returned as captions.
        """
        logging.error(f"collection_name.value:{collection_name}, dataset_dir:{self.local_metric_datasets_dir}")
        local_collection_dir = Path(f"{self.local_metric_datasets_dir}/{collection_name.value}")
        logging.error(f"collection_name.value:{collection_name.value}, local_collection_dir:{local_collection_dir}")
        captions_local = random.choices(list(local_collection_dir.iterdir()), k=k*2)
        logging.info(f"captions_local:{captions_local}")
        imgs_local = []
        captions_local_str = []
        for caption in captions_local:
            if ".market." in caption.name:
                continue
            logging.info(f"caption:{caption}")
            imgs_local.append(Image.open(f"{caption}"))
            captions_local_str.append(
                caption.name
            )  # this result is loaded directly to the application state
        return captions_local_str, imgs_local

    def get_meta_json(
        self, collection_name: MetricCollections
    ) -> dict:
        """
        Get meta.json dictionary created during model training from local storage.
        """
        with open(f"{self.local_models_dir}/{collection_name.value}/meta.json") as f:
            return json.load(f)
