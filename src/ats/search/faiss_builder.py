import logging
import os
import csv
import pickle
import time
import faiss
import numpy as np

#from kaleido.scopes.plotly import PlotlyScope
import plotly.graph_objects as go
from pytorch_forecasting.utils import detach, to_list
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from ats.prediction import prediction_data
from ats.prediction import prediction_utils
from ats.model import viz_utils
#import plotly.io as pio
#pio.orca.config.use_xvfb = True

#import plotly.io as pio
#pio.kaleido.scope.mathjax = None

class FaissBuilder(object):
    def __init__(self, env_mgr, model, market_data_mgr, wandb_logger):
        super().__init__()
        self.env_mgr = env_mgr
        self.model = model
        self.wandb_logger = wandb_logger
        self.config = env_mgr.config
        self.data_module = market_data_mgr.data_module()
        self.num_samples = self.config.job.eval_batches
        self.eval_batch_size = self.config.model.eval_batch_size
        self.every_n_epochs = self.config.job.log_example_eval_every_n_epochs
        self.embedding_cache_path = self.config.job.embedding_cache_path
        self.image_root_path = self.config.job.image_root_path + "/" + self.env_mgr.run_id
        os.makedirs(self.image_root_path, exist_ok=True)
        self.embedding_size = 768  # Size of embeddings
        self.top_k_hits = 10  # Output k hits
        logging.info(f"num_samples:{self.num_samples}, eval_batch_size={self.eval_batch_size}")
        # Defining our FAISS index
        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        self.n_clusters = 1024

        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.quantizer = faiss.IndexFlatIP(self.embedding_size)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.embedding_size, self.n_clusters, faiss.METRIC_INNER_PRODUCT
        )

        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
        self.index.nprobe = 3

        self.validation = self.data_module.validation
        self.matched_eval_data = self.data_module.eval_data

    def create_market_image(self, pred_input, pred_output):
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
            ],
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            yaxis=dict(
                side="right",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
            ],
        )
        fig.update_layout(title=pred_input.prediction_date_time, font=dict(size=20))
        y_max, y_min, y_hat_max, y_hat_min = prediction_utils.loss_stats(pred_output)
        viz_utils.add_market_viz(fig, pred_input)
        decoder_time_idx = pred_input.decoder_time_idx
        #logging.info(f"after add_market_viz")
        output_file = f'{self.image_root_path}/{decoder_time_idx}_{pred_input.prediction_date_time}_{y_max}_{y_min}_{y_hat_max}_{y_hat_min}.market.png'
        output_file = output_file.replace(" ","_")
        try:
            fig.write_image(output_file)
            logging.info(f"generate market_image {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"can not generate {output_file}, {e}")
            return None

    def create_image(self, pred_input, pred_output):
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
                [
                    {"secondary_y": True},
                    {"secondary_y": True},
                    {"secondary_y": True},
                ],
            ],
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            yaxis=dict(
                side="right",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
            ],
        )
        fig.update_layout(title=pred_input.prediction_date_time, font=dict(size=20))
        #viz_utils.add_market_viz(fig, pred_input)
        viz_utils.add_model_prediction(fig, self.model, pred_input, pred_output)
        viz_utils.add_model_interpretation(fig, self.model, pred_input, pred_output)
        decoder_time_idx = pred_input.decoder_time_idx
        viz_utils.add_market_viz(fig, pred_input)
        y_max, y_min, y_hat_max, y_hat_min = prediction_utils.loss_stats(pred_output)
        output_file = f'{self.image_root_path}/{decoder_time_idx}_{pred_input.prediction_date_time}_{y_max}_{y_min}_{y_hat_max}_{y_hat_min}.png'
        output_file = output_file.replace(" ","_")
        try:
            fig.write_image(output_file)
            logging.info(f"generate image {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"can not generate {output_file}, {e}")
            return None

    def build_embedding_cache(self):
        device = self.model.device
        data_iter = iter(self.data_module.val_dataloader())

        corpus_images = list()
        corpus_embeddings = list()

        for batch in range(self.num_samples):
            #logging.info(f"batch:{batch}")
            val_x, val_y = next(data_iter)
            if self.config.job.eval_batch_start_idx>-1 and batch<self.config.job.eval_batch_start_idx:
                continue
            if self.config.job.eval_batch_end_idx>-1 and batch>self.config.job.eval_batch_end_idx:
                break
            y_close = val_y[0]
            logging.info(f"y_close:{y_close.shape}")
            y_close_cum_sum = torch.cumsum(y_close, dim=-1)
            # indices are based on decoder time idx (first prediction point)
            indices = self.data_module.validation.x_to_index(val_x)
            decoder_time_idx = indices.time_idx
            #logging.info(f"decoder_time_idx:{decoder_time_idx}")
            filtered_dataset = self.validation.filter(
                lambda x: x.time_idx_first_prediction.isin(decoder_time_idx)
            )
            y_hats, y_quantiles, output, ret_x = prediction_utils.predict(
                self.model,
                filtered_dataset,
                self.wandb_logger,
                batch_size=self.eval_batch_size,
            )
            interp_output = self.model.interpret_output(
                detach(output),
                reduction="none",
                attention_prediction_horizon=0,  # attention only for first prediction horizon
            )
            logging.info(f"y_hats:{len(y_hats)}")
            for idx in range(y_hats.size(0)):
                index = indices.iloc[idx]
                pred_input = prediction_data.PredictionInput(
                    x=ret_x,
                    idx=idx
                )
                prediction_utils.add_pred_context(self.env_mgr, self.matched_eval_data, idx, index, pred_input)
                pred_output = prediction_data.PredictionOutput(
                    out=output,
                    idx=idx,
                    y_close_cum_sum=y_close_cum_sum,
                    y_hats=y_hats,
                    y_quantiles=y_quantiles,
                    interp_output=interp_output,
                    embedding=output["embedding"],
                )
                embedding = pred_output.embedding[idx]
                image = self.create_image(pred_input, pred_output)
                market_image = self.create_market_image(pred_input, pred_output)
                corpus_embeddings.append(embedding)
                corpus_images.append({"model":image, "market":market_image})

        logging.info("Store file on disc")
        base_dir = os.path.dirname(self.embedding_cache_path)
        os.makedirs(base_dir, exist_ok=True)
        with open(self.embedding_cache_path, "wb") as fOut:
            pickle.dump(
                {"images": corpus_images, "embeddings": corpus_embeddings}, fOut
            )

    def load_index(self):
        ### Create the FAISS index
        print("Start creating FAISS index")
        # First, we need to normalize vectors to unit length
        corpus_embeddings = (
            corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        )

        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)

        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)

    def search(
        self,
    ):
        inp_question = input("Please enter a question: ")

        start_time = time.time()
        question_embedding = model.encode(inp_question)

        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = index.search(question_embedding, top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [
            {"corpus_id": id, "score": score}
            for id, score in zip(corpus_ids[0], distances[0])
        ]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        end_time = time.time()

        print("Input question:", inp_question)
        print("Results (after {:.3f} seconds):".format(end_time - start_time))
        for hit in hits[0:top_k_hits]:
            print(
                "\t{:.3f}\t{}".format(hit["score"], corpus_sentences[hit["corpus_id"]])
            )

    def build_embedding_cache_if_not_exists(self):
        # Check if embedding cache path exists
        #if not os.path.exists(self.embedding_cache_path):
        self.build_embedding_cache()
        #else:
        #if True:
        #    print("Load pre-computed embeddings from disc")
        #    with open(self.embedding_cache_path, "rb") as fIn:
        #        cache_data = pickle.load(fIn)
        #        corpus_images = cache_data["images"]
        #        corpus_embeddings = cache_data["embeddings"]
                #logging.info(f"corpus_images:{corpus_images}")
                #logging.info(f"corpus_embeddings:{corpus_embeddings}")
