import logging
from typing import Any

import streamlit as st
from loguru import logger
from PIL import Image
import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components

from vss.common import env_handler
from vss.common.consts import CATEGORY_DESCR, GRID_NROW_NUMBER, INTERACTIVE_ASSETS_DICT
from vss.metrics.consts import MetricCollections
from vss.metrics.core import BestChoiceImagesDataset, MetricClient


#@st.cache_resource
def get_metric_client(_cfg):
    return MetricClient(cfg=_cfg)

class ModuleManager:
    """
    List of components used for building the app.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def widget_formatting(self) -> Any:
        """
        Defines Streamlit widget styles based on the input provided by style.css file.
        """
        with open(INTERACTIVE_ASSETS_DICT["widget_style_file"], "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def initialize_states(self) -> None:
        """
        Initializes states within the application. Those are distinct for every user.
        The "if" condition is necessary for the app to not reset it's whole session_state.
        """
        if "init" not in st.session_state:
            st.session_state.init = True
            st.session_state.metric_client = get_metric_client(self.cfg)
            st.session_state.category_desc_option = None
            st.session_state.category_option = MetricCollections.FUTURES
            st.session_state.provisioning_options = None

            # Option 1 States - Example Images
            st.session_state.example_captions = None
            st.session_state.example_imgs = None
            st.session_state.example_list_pull = False  # NEW

            # Option 2 States - Storage Images
            st.session_state.pull_random_img_number = None
            st.session_state.refresh_random_images = None
            st.session_state.random_captions = None
            st.session_state.random_imgs = None
            st.session_state.img_storage_list_pull = False  # NEW

            # Option 3 States - Uploaded File Images
            st.session_state.file_upload_pull = False  # NEW

            # All Options States - set when an input image has been selected
            st.session_state.show_input_img = None
            st.session_state.selected_img = None

            # Search and Output States - set when an input image has been selected and before "Find Similar Images" is run
            st.session_state.similar_img_number = None
            st.session_state.benchmark_similarity_value = None
            st.session_state.grid_nrow_number = GRID_NROW_NUMBER

            # Search Completed State - set after "Find Similar Images" is completed
            #st.session_state.similar_images_found = None

    def reset_all_states_button(self) -> None:
        """
        Reset all application starting from category selection.
        """
        st.session_state.category_desc_option = None
        st.session_state.category_option = None
        st.session_state.provisioning_options = None
        st.session_state.show_input_img = None
        st.session_state.selected_img = None
        st.session_state.benchmark_similarity_value = None
        st.session_state.similar_img_number = None
        #st.session_state.similar_images_found = None

        st.session_state.example_list_pull = None  # NEW
        st.session_state.img_storage_list_pull = None  # NEW
        st.session_state.file_upload_pull = None  # NEW

    def reset_states_after_image_provisioning_list(self) -> None:
        """
        Reset all application states after radio selection for image provisioning.
        """
        st.session_state.show_input_img = None
        st.session_state.selected_img = None
        st.session_state.benchmark_similarity_value = None
        st.session_state.similar_img_number = None
        #st.session_state.similar_images_found = None

    def make_grid(self, cols, rows) -> Any:
        grid = [0] * cols
        for i in range(cols):
            with st.container():
                grid[i] = st.columns(rows)
        return grid

    def create_main_filters(self) -> None:
        """
        Adds initial header, reset all filers button, and category selection buttons.
        """
        st.markdown(
            "<h2 style='text-align: center;'>Input Options</h2>", unsafe_allow_html=True
        )
        with st.expander("", expanded=True):
            st.markdown(
                "<h3 style='text-align: center;'>Which category would you like to search from?</h3>",
                unsafe_allow_html=True,
            )
            mygrid = self.make_grid(4, (2, 2, 2, 2))
            for idx, category_enum in enumerate(MetricCollections):
                logging.info(f"add option: {category_enum}")
                with mygrid[idx // 2][1 + idx % 2]:
                    if st.button(
                        CATEGORY_DESCR[category_enum.value]["description"]
                    ):  # category buttons
                        st.session_state.category_desc_option = CATEGORY_DESCR[
                            category_enum.value
                        ]["description"]
                        st.session_state.category_option = category_enum

            #if st.session_state.category_option:
            #    st.markdown(
            #        "<h3 style='text-align: center;'>Business Cases</h3>",
            #        unsafe_allow_html=True,
            #    )
            #    st.write(
            #        CATEGORY_DESCR[st.session_state.category_option.value][
            #            "business_usage"
            #        ]
            #    )
            #    st.write(
            #        f"Source Dataset: [link]({CATEGORY_DESCR[st.session_state.category_option.value]['source']})"
            #    )

    def create_image_provision_for_examples(self) -> None:
        """
        Resets state of previous provisioning selection and creates a category-specific list of image examples
        that a user can select from.
        """
        if st.session_state.example_list_pull is True:
            self.reset_states_after_image_provisioning_list()

        st.session_state.example_captions = [
            # Need to use path so that we can get example key
            s_img["path"]
            for s_img in CATEGORY_DESCR[st.session_state.category_option.value][
                "image_examples"
            ]
        ]  # get captions
        st.session_state.example_imgs = [
            Image.open(s_img["path"])
            for s_img in CATEGORY_DESCR[st.session_state.category_option.value][
                "image_examples"
            ]
        ]  # get images
        example_images_zip = dict(
            zip(st.session_state.example_captions, st.session_state.example_imgs)
        )
        img_selection = st.selectbox(
            f"Choose an image - {st.session_state.category_option.value}.",
            example_images_zip,
        )  # select image

        if st.session_state.example_list_pull is True:
            st.session_state.selected_img = example_images_zip[img_selection]
            logging.info(f"st.session_state.selected_img:{st.session_state.selected_img}")
            st.session_state.show_input_img = True

    def create_image_provision_for_random_storage_pull(self) -> None:
        """
        Resets state of previous provisioning selection and pulls from local/cloud storage a category-specific list of
        image examples that a user can select from the list. Additionally, a button for re-running random selection is
        implemented together with the input option for the number of sampled images.
        """
        st.session_state.img_storage_list_pull = True  # TODO: Upload options other than storage pull temporarily turned off, remove this line when turning on
        if st.session_state.img_storage_list_pull is True:
            self.reset_states_after_image_provisioning_list()
        st.session_state.pull_random_img_number = st.number_input(
            label=f"Choose random images",
            value=5,
            min_value=1,
            format="%i",
        )
        #if st.button("Generate Images"):
        if True:
            try:
                (
                    st.session_state.random_captions,
                    st.session_state.random_imgs,
                ) = env_handler.get_random_images_from_collection(
                    collection_name=st.session_state.category_option,
                    #collection_name=MetricCollections.FUTURES,
                    k=st.session_state.pull_random_img_number,
                )  # Pulls a sampled set of images from local/cloud storage
            except Exception as e:
                logging.error(f"can not get random images: {e}")
        logging.error(f"st.session_state.random_captions:{st.session_state.random_captions}");
        #if st.session_state.random_captions and st.session_state.random_imgs:
        if True:
            random_images_zip = dict(
                zip(
                    st.session_state.random_captions,
                    st.session_state.random_imgs,
                )
            )
            logging.error(f"random_images_zip:{random_images_zip}")
            img_selection = st.selectbox("Choose an image", random_images_zip)
            logging.error(f"img_selection:{img_selection}")
            #if st.session_state.img_storage_list_pull is True:
            if True:
                st.session_state.selected_img_path = img_selection
                st.session_state.selected_img = random_images_zip[
                    img_selection
                ]  # returns an image based on selection
                logging.error(f"st.session_state.selected_img:{st.session_state.selected_img}")
                logging.error(f"st.session_state.selected_img_path:{st.session_state.selected_img_path}")
                st.session_state.show_input_img = True

    def create_similarity_search_filters(self) -> None:
        """
        Creates a set of similarity-search-specific filters - number of shown images an benchmark for
        minimum similarity value in %.
        """
        if True:
        #if st.session_state.show_input_img:
            st.markdown(
                "<h2 style='text-align: center;'>Input Image</h2>",
                unsafe_allow_html=True,
            )

            with st.expander("", expanded=True):
                col_search_1, col_search_2 = st.columns([1, 1])
                with col_search_1:
                    st.image(st.session_state.selected_img)
                with col_search_2:
                    st.markdown(
                        "<h4 style='text-align: center;'>Search Options</h4>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.similar_img_number = st.number_input(
                        label="Insert a number of similar images to show.",
                        value=9,
                        min_value=1,
                        format="%i",
                    )
                    st.session_state.benchmark_similarity_value = st.number_input(
                        label="Insert a benchmark similarity value (in %).",
                        value=50,
                        min_value=0,
                        max_value=100,
                        format="%i",
                    )

                    logger.info("Similarity Search Button")
                    button = st.button(
                        "Find Similar Images", key="similar_images_found"
                    )
                    logger.info("After setting button")

    def search_with_show(
        self,
        collection_name: MetricCollections,
        k: int,
        grid_nrow: int,
        benchmark: int,
        key: str,
        file,
    ) -> None:
        """
        Shows images in order of their similarity to the original input image.
        """
        metric_client = st.session_state.metric_client
        best_images_dataset = (
            BestChoiceImagesDataset.get_best_choice_for_uploaded_image(
                client=metric_client,
                key=key,
                anchor=file,
                collection_name=collection_name,
                k=k,
                benchmark=benchmark,
            )
        )
        self.df = pd.DataFrame(columns=best_images_dataset.results[0].payload.keys())
        for r in best_images_dataset.results:
            self.df = self.df.append(r.payload, ignore_index = True)
        logging.error(f"df:{self.df}")
        captions_dict = [
            {
                "file": r.payload["file"].split("/")[-1].split("\\")[-1],
                "class": r.payload["ticker"],
                "similarity": "{0:.2f}%".format(100 * round(r.score, 4)),
            }
            for r in best_images_dataset.results
        ]
        if best_images_dataset.similars:
            with st.expander("", expanded=True):
                results_text = f'Found {st.session_state.similar_img_number} images in the "{st.session_state.category_option.value}" category'
                st.markdown(
                    f"<h4 style='text-align: center;'>{results_text}</h4>",
                    unsafe_allow_html=True,
                )
                comment_text = f"In the top left corner of every image a similarity coefficient is presented - it shows a level of similarity between a given image and an input image."
                st.markdown(
                    f"<p style='text-align: center;'>{comment_text}</p>",
                    unsafe_allow_html=True,
                )
                col_nr = min(grid_nrow, len(best_images_dataset.similars))
                for i, col in enumerate(st.columns(col_nr)):
                    col_imgs = best_images_dataset.similars[i::col_nr]
                    col_imgs_captions_dict = captions_dict[i::col_nr]
                    with col:
                        for j, col_img in enumerate(col_imgs):
                            st.image(col_img)
                            #st.markdown(
                            #    f"<p style='text-align: start; color: red; font-weight: bold;'>Similarity: {col_imgs_captions_dict[j]['similarity']}</p>",
                            #    unsafe_allow_html=True,
                            #)
                            #st.markdown(
                            #    f"<p style='text-align: start; font-weight: bold;'>Class: {col_imgs_captions_dict[j]['class']}</p>",
                            #    unsafe_allow_html=True,
                            #)
                            #st.markdown(
                            #    f"<p style='text-align: start; font-weight: bold;'>File: {col_imgs_captions_dict[j]['file']}</p>",
                            #    unsafe_allow_html=True,
                            #)
                            #st.write("")
                            #st.write("")

        else:
            no_results_text = st.write(
                f"No images found for the similarity benchmark of {benchmark}%."
            )
            st.markdown(
                f"<h4 style='text-align: center; color: red;'>{no_results_text}</h4>",
                unsafe_allow_html=True,
            )

    def extract_similar_images(self) -> None:
        """
        Shows images in order of their similarity to the original input image.
        """
        logging.error(f"st.session_state.category_option:{st.session_state.category_option}")
        logging.error(f"session_state:{st.session_state}")
        self.search_with_show(
            key=st.session_state.selected_img_path,
            file=st.session_state.selected_img,
            collection_name=st.session_state.category_option.value,
            k=st.session_state.similar_img_number,
            grid_nrow=st.session_state.grid_nrow_number,
            benchmark=st.session_state.benchmark_similarity_value,
        )

        if st.button("Reset Images"):
            self.reset_all_states_button()

    def create_image_load(self) -> None:
        """"""
        st.markdown(
            "<h2 style='text-align: center;'>Upload Options</h2>",
            unsafe_allow_html=True,
        )
        with st.expander("", expanded=True):
            st.markdown(
                "<h3 style='text-align: center;'>How would you like to add an image?</h3>",
                unsafe_allow_html=True,
            )

            col_image_load_1, col_image_load_2, col_image_load_3 = st.columns(3)

            with col_image_load_2:
                try:
                    self.create_image_provision_for_random_storage_pull()
                except Exception as e:
                    logging.error(f"can not get random images:{e}")

    def run_app(self) -> None:
        try:
            # Check if 'key' already exists in session_state
            # If not, then initialize it
            if 'key' not in st.session_state:
                st.session_state['key'] = 'value'

            logger.info("Set main graphical options.")            
            st.set_page_config(page_title="visual-search.stxnext.pl", layout="wide")
            self.widget_formatting()

            logger.info("Initialize states.")
            self.initialize_states()

            #logger.info("Create Main Filters - till category search")
            #self.create_main_filters()

            logger.info("Image Provisioning")
            self.create_image_load()

            if st.session_state.category_option and st.session_state.selected_img:
                logger.info("Similarity Search Filters")
                self.create_similarity_search_filters()
                #logging.info(f"st.session_state.similar_images_found:{st.session_state.similar_images_found}")
                #if st.session_state.similar_images_found:
                self.extract_similar_images()

            # Import your data
            df = self.df
 
            # Generate the HTML using Pygwalker
            pyg_html = pyg.walk(df, return_html=True)
 
            # Embed the HTML into the Streamlit app
            components.html(pyg_html, height=1000, scrolling=True)


        except Exception as e:
            logging.error(f"can not run app {e}")
