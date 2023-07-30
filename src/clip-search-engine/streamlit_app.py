import streamlit as st
import pinecone
from utils import get_text_embedding


PINECONE_KEY = st.secrets["PINECONE_KEY"]

st.markdown(
    """
<style>
#introduction {
    padding: 0px 20px 0px 20px;
    background-color: #ffffd9;
    border-radius: 10px;

}

#introduction p {
    font-size: 1.1rem;
    color: #a17112;

}

img {
    padding: 5px;
}
</style>


""",
    unsafe_allow_html=True,
)

st.markdown("# CLIP Image Search Engine")

st.markdown(
    """
<div id="introduction">

<p>
Hello! This is a demo of a Pinecone-powered image search engine using the CLIP model. 
The dataset used is the COCO 2017 dataset, which contains 118,287 images. 
To get started, type in a search query and simply press enter. 
Try searching for objects more than people, as COCO is a dataset of objects.

Happy searching!

</p>
</div>
""",
    unsafe_allow_html=True,
)

pinecone.init(api_key=PINECONE_KEY, environment="us-east4-gcp")  # app.pinecone.io

index_name = "clip-image-search"
index = pinecone.Index(index_name=index_name)

text_query = st.text_input(":mag_right: Search for images", "salad on a plate")

number_of_results = st.slider("Number of results ", 1, 100, 10)

query_vector = get_text_embedding(text_query)

top_k_samples = index.query(
    vector=query_vector, top_k=number_of_results, include_values=False
)

st.markdown("<div style='align: center; display: flex'>", unsafe_allow_html=True)
st.image([str(result.id) for result in top_k_samples["matches"]], width=230)
st.markdown("</div>", unsafe_allow_html=True)
