import torch
import logging
import numpy as np
from utils import Pipeline
from models import (
    AttentionLSTM,
    VanillaLSTM,
    EmbeddingLSTM,
    AttentionEmbeddingLSTM
)
from data_module import AtsDataModule

torch.manual_seed(0)
np.random.seed(0)


class AttentionLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        X_train = self.X_train
        y_train = self.y_train

        model = AttentionLSTM(
            embed_dim=X_train.shape[2], out_size=y_train.shape[-1]
        )
        self.model = model


class VanillaLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        X_train = self.X_train
        y_train = self.y_train

        model = VanillaLSTM(
            input_size=X_train.shape[2], out_size=y_train.shape[-1]
        )
        self.model = model


class EmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()
        self.dataset = dataset

    def create_model(self):
        X_train = self.X_train
        y_train = self.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]

        model = EmbeddingLSTM(
            linear_channel=features,
            period_channel=(mini_batch - features),
            input_channel=mini_batch,
            input_size=X_train.shape[2],
            out_size=y_train.shape[-1]
        )
        self.model = model


class AttentionEmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()
        self.dataset = dataset

    def create_model(self):
        self.data_module = AtsDataModule("stock_returns")
        X_train = self.data_module.X_train
        y_train = self.data_module.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]

        #device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        #device = "cpu"
        #DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #logging.info(f"device:{device}")
        model = AttentionEmbeddingLSTM(
            linear_channel=features,
            period_channel=(mini_batch - features),
            input_channel=mini_batch,
            input_size=X_train.shape[2],
            out_size=y_train.shape[-1]
        )
        self.model = model
