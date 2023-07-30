# Dataset tools & annotation workflow

## Tools
- Duplicate image finder: search one or more datasets efficiently for duplicate images: [duplicate_image_finder.ipynb](duplicate_image_finder.ipynb)
  - Logs for COCO, MPII, OCHUman here: [data/logs](data/logs)
- Annotation to COCO conversion (human keypoint annotations): [Datasets_to_COCO.ipynb](Datasets_to_COCO.ipynb)
  - MPII, CrowdPose, OCHuman, download here: [data/anno_files](data/anno_files)
  

## CVAT-fiftyone annotation workflow

1. Start [CVAT](https://github.com/openvinotoolkit/cvat) in Docker
2. Create .env file with CVAT credentials (see .env.example)
3. Start a cvat_*.ipynb notebook

Paths to datasets can be added/adjusted in configs/default.py.

**Note**: CVAT does currently not support labeled keypoints (e.g., COCO format).

## coco-annotator annotation workflow

Check:
- [coco-annotator-setup.md](docs/coco-annotator-setup.md) for working with the coco annotator
- [human_keypoint_annotation.md](docs/human_keypoint_annotation.md) for a keypoint-annotation guideline

