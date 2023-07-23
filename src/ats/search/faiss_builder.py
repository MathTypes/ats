from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import faiss
import numpy as np


class FaissBuilder(object):
    def __init__(self, env_mgr, model, market_data_mgr, wandb_logger):
        super().__init__()
        this.env_mgr = env_mgr
        this.model = model
        this.config = env_mgr.config
        this.data_module = market_data_mgr.data_module
        self.val_x_batch = []
        self.val_y_batch = []
        self.indices_batch = []
        self.num_samples = self.config.job.eval_batches
        self.every_n_epochs = self.config.job.log_example_eval_every_n_epochs
        self.embedding_cache_path = self.config.job.embedding_cache_path
        self.embedding_size = 768  # Size of embeddings
        self.top_k_hits = 10  # Output k hits
        logging.info(f"num_samples:{self.num_samples}")
        # Defining our FAISS index
        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        self.n_clusters = 1024

        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.quantizer = faiss.IndexFlatIP(embedding_size)
        self.index = faiss.IndexIVFFlat(
            quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT
        )

        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
        self.index.nprobe = 3

        data_iter = iter(data_module.val_dataloader())
        for batch in range(self.num_samples):
            val_x, val_y = next(data_iter)
            indices = data_module.validation.x_to_index(val_x)
            self.val_x_batch.append(val_x)
            self.val_y_batch.append(val_y)
            self.indices_batch.append(indices)
            logging.info(
                f"batch_size:{len(val_x)}, indices_batch:{len(self.indices_batch)}"
            )
        self.validation = data_module.validation
        self.matched_eval_data = data_module.eval_data

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
            width=1500,
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
        viz_utils.add_market_viz(fig, train_data_rows)
        viz_utils.add_model_prediction(fig, self.model, pred_input, pred_output)
        viz_utils.add_model_interpretation(fig, self.model, pred_input, pred_output)
        img_bytes = fig.to_image(format="png")  # kaleido library
        im = PIL.Image.open(BytesIO(img_bytes))
        img = wandb.Image(im)

    def build_embedding_cache(self):
        device = self.pl_module.device
        data_iter = iter(self.data_module.eval_dataloader())

        corpuse_images = list()
        corpus_embeddinds = list()

        for batch in range(self.num_samples):
            val_x, val_y = next(data_iter)
            indices = self.data_module.validation.x_to_index(val_x)
            logging.info(f"indices:{indices}")
            filtered_dataset = self.train_dataset.filter(
                lambda x: (x.time_idx_last % self.sample_n == 0)
            )
            y_hats, y_quantiles, output = prediction_utils.predict(
                self.module,
                filtered_dataset,
                self.wandb_logger,
                batch_size=self.num_samples,
            )
            interp_output = self.pl_module.interpret_output(
                detach(output),
                reduction="none",
                attention_prediction_horizon=0,  # attention only for first prediction horizon
            )
            for idx in range(len(y_hats)):
                pred_input = prediction_data.PredictionInput(
                    x=output.x, idx=idx, train_data_rows=train_data_rows
                )
                pred_output = prediction_data.PredictionOutput(
                    out=output,
                    idx=idx,
                    y_hats=out.y_hats,
                    y_quantiles=out.y_quantiles,
                    interp_output=interp_output,
                    embedding=output.embedding,
                )
                corpus_embeddinds.append(pred_output.embedding)
                corpus_images.append(self.create_image(pred_input, pred_output))
        print("Store file on disc")
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
        if not os.path.exists(self.embedding_cache_path):
            build_embedding_cache()
        else:
            print("Load pre-computed embeddings from disc")
            with open(embedding_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_sentences = cache_data["sentences"]
                corpus_embeddings = cache_data["embeddings"]
