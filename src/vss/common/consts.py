import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent)
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
MINIO_MAIN_PATH = Path(os.getenv("MINIO_MAIN_PATH"))
INTERACTIVE_ASSETS_DICT = {
    "logo_img_dir": f"{PROJECT_PATH}/interactive/assets/logo_stx.jpg",
    "widget_style_file": f"{PROJECT_PATH}/interactive/widget_style.css",
    "app_title": "Visual Similarity Search Engine",
    "app_first_paragraph": """
            Returns a set number of images from a selected category. 
            This set contains images with the highest degree of similarity to the uploaded/selected image.
            Returned images are pulled from the local or cloud storage and similarity is calculated based on the vectors 
            stored in the Qdrant database.
            Algorithm uses image embeddings and deep neural networks to determine a value of cosine similarity metric.
        """,
    "github_link": "https://github.com/stxnext/visual-similarity-search",
    "our_ml_site_link": "https://www.stxnext.com/services/machine-learning/?utm_source=github&utm_medium=mlde&utm_campaign=visearch-demo",
}
GRID_NROW_NUMBER = 3
EXAMPLE_PATH = f"{PROJECT_PATH}/interactive/examples"
CATEGORY_DESCR = {
    "futures": {
        "description": "Futures market.",
        "business_usage": """
            Dogs are just an example and following use cases can be extrapolated to other animal species. \n
            **USE CASE #1**: Identifying a breed of the dog based on its picture. 
            It may be useful for veterinarians and other animal-specialized occupations. \n
            **USE CASE #2**: Finding other breeds based on the similarity factor. 
            It may be useful for breeders and people looking to buy an animal.
        """,
        "source": "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset",
        "bootstrap_icon": "bag",
        "image_examples": [
            {
                "id": "dog_1",
                "path": f"{EXAMPLE_PATH}/dogs_img_1.png",
                "label": "Cane Corso",
            },
            {
                "id": "dog_2",
                "path": f"{EXAMPLE_PATH}/dogs_img_2.jpg",
                "label": "St. Bernard",
            },
        ],
    },
}
