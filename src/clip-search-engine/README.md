# CLIP-powered Semantic Search Engine using Pinecone, FiftyOne, and Streamlit

This repository will allow you to generate text embeddings for the COCO2017 dataset, as well as deploy a Streamlit webapp search engine on your local machine. 

If you want to check out the deployed app, click [here](https://clip-search-engine.streamlit.app/). The app works best on Safari, Mozilla, or DuckDuckGo. You might experience bugs on Chromium-based browsers.

## Uploading the Data on your Pinecone
1. Clone this repository

2. Run `pip install -r requirements.txt`

3. Make sure you have a Pinecone account and API Key. You can find this on your [console](https://app.pinecone.io/). 

4. Create a file in your directory called `config.py`, in which you create a variable called `PINECONE_KEY = your_key_here`

5. Run the `data_upload.ipynb` notebook

Great! You should have an Index now on your Pinecone console, called `clip-image-search`.

## Running the webapp locally
1. Install Streamlit on your machine if you don't have it

2. In the .streamlit folder, create a `secrets.toml` file

3. Insert your Pinecone key, such as `PINECONE_KEY = your_key_here`

4. Run `streamlit run streamlit_app.py`

The app should be on your localhost! Have fun searching :)

