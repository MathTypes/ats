import numpy as np
import os
import pandas as pd
import pydeck
import streamlit as st

data_root_processed = 'data'


@st.cache(persist=False, allow_output_mutation=False)
# @st.cache(persist=True)
def load_data(input_json_data):
    data = pd.read_json(input_json_data)
    return data


st.title("Analyzing RSS News Feeds")
st.markdown("This is a dashboard for analyzing and visualizing headlines from RSS news feeds "
            "of some of the popular news providers on the web")

# List of news sources we have
display_source = ('bbc', 'cnn', 'dailystar', 'guardian', 'ht', 'jpost', 'khaleejtimes',
                  'npr', 'nyt', 'pd', 'rt', 'thehindu', 'toi', 'wp', 'xinhua', 'yahoonews')

options = list(range(len(display_source)))
sel = st.selectbox("News source:", options,
                   format_func=lambda x: display_source[x])
source = display_source[sel]
target = os.path.join(data_root_processed, source+'.json')
data = load_data(target)
unmodified_data = data

st.header(
    f"News headlines from {source} between {data['date_time'].min()} and {data['date_time'].max()}")

st.header(
    f"Which locations in the world are connected to the news headlines from {source} ?")
st.map(data.query("source == @source")[["latitude", "longitude"]], zoom=0.7)

sentiment_dict = {'Negative': 0, 'Positive': 1}
sentiment_tup = tuple(sentiment_dict)
sentiment_options = list(range(len(sentiment_tup)))
sentiment_sel = st.selectbox("News sentiment:", sentiment_options,
                             format_func=lambda x: sentiment_tup[x])
data = unmodified_data[unmodified_data['sentment'] == sentiment_sel]
st.header(
    f"From where are the news headlines with {sentiment_tup[sentiment_sel]} sentiment being reported by {source} ?")

# Calculate average latitude and longitude of all the locations so that map can be centered
map_center = (np.average(data['latitude']), np.average(data['longitude']))

# Define a layer to display on a map
layer = pydeck.Layer(
    'HexagonLayer',
    data,
    get_position=['longitude', 'latitude'],
    radius=70000,
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1)

# Set the viewport location
view_state = pydeck.ViewState(
    longitude=map_center[1],
    latitude=map_center[0],
    zoom=2,
    min_zoom=1,
    max_zoom=15,
    pitch=50,
    bearing=0.0)

# Streamlit render
st.write(pydeck.Deck(layers=[layer], initial_view_state=view_state))

# Display unmodified data if checkbox is selected
if st.checkbox("Show Map Data", False):
    st.subheader('Map Data')
    st.write(unmodified_data)
