"""Default project config.
"""

import os

from dotenv import load_dotenv
from yacs.config import CfgNode as CN

load_dotenv()
_C = CN()

# Set environment variables
os.environ['FIFTYONE_CVAT_USERNAME'] = os.getenv('FIFTYONE_CVAT_USERNAME')
os.environ['FIFTYONE_CVAT_PASSWORD'] = os.getenv('FIFTYONE_CVAT_PASSWORD')
# ... to cfg
_C.CRED = CN()
_C.CRED.FIFTYONE_CVAT_USERNAME = os.getenv('FIFTYONE_CVAT_USERNAME')
_C.CRED.FIFTYONE_CVAT_PASSWORD = os.getenv('FIFTYONE_CVAT_PASSWORD')
_C.CRED.FLICKR_KEY = os.getenv('FLICKR_KEY')
_C.CRED.FLICKR_SECRET = os.getenv('FLICKR_SECRET')


# Datasets
_C.DATASET = CN()
# ROOT/annotations/[file] and ROOT/images/[coco_set]/(images)
_C.DATASET.COCO = CN()
_C.DATASET.COCO.ROOT = 'data/coco'
# ROOT/json/[file] and ROOT/[images]/(images)
_C.DATASET.CROWD_POSE = CN()
_C.DATASET.CROWD_POSE.ROOT = 'data/crowd_pose'
# ROOT/annotations/[file] and ROOT/[images]/(images)
_C.DATASET.MYIMG = CN()
_C.DATASET.MYIMG.ROOT = 'data/fireground_pose'


# Tools
_C.FIFTYONE, _C.CVAT, _C.COCO_ANNOTATOR = CN(), CN(), CN()
_C.FIFTYONE.PORT = 5151  # app default is 5151
_C.FIFTYONE.URL = f'http://localhost:{_C.FIFTYONE.PORT}/'
_C.CVAT.PORT = 8080  # app default is 8080
_C.CVAT.URL = f'http://localhost:{_C.CVAT.PORT}/'
_C.COCO_ANNOTATOR.PORT = 5001  # app default is 5000
_C.COCO_ANNOTATOR.URL = f'http://localhost:{_C.COCO_ANNOTATOR.PORT}/'
