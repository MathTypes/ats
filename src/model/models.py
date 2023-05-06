import statistics
import torch
import torch.nn as nn
from utils import Time2Vec
import wandb
import pytorch_lightning as pl
torch.manual_seed(0)
from torch.nn import functional as F

class AttentionLSTM(pl.LightningModule):
    """
    Multihead-attention model + LSTM layer.
    """
    def __init__(self, embed_dim, out_size, hidden_size=17, n_layers=2):
        super(AttentionLSTM, self).__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, weights = self.att(X, X, X)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out


class VanillaLSTM(pl.LightningModule):
    """
    Multihead-attention model + LSTM layer.
    """
    def __init__(self, input_size, out_size, hidden_size=20, n_layers=2):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, (h, c) = self.lstm(X)
        out = self.lin(out)
        return out


class EmbeddingLSTM(pl.LightningModule):
    """
    Time2vec embedding + LSTM.
    """
    def __init__(
        self,
        linear_channel,
        period_channel,
        input_channel,
        input_size,
        out_size,
        hidden_size=19,
        n_layers=2
    ):
        super(EmbeddingLSTM, self).__init__()
        self.emb = Time2Vec(linear_channel, period_channel, input_channel)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out = self.emb(X)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out


class AttentionEmbeddingLSTM(pl.LightningModule):
    """
    Time2vec embedding + Attention + LSTM.
    """
    def __init__(
        self,
        linear_channel,
        period_channel,
        input_channel,
        input_size,
        out_size,
        hidden_size=16,
        n_layers=2
    ):
        super(AttentionEmbeddingLSTM, self).__init__()
        self.criterion = nn.HuberLoss()
        self.emb = Time2Vec(linear_channel, period_channel, input_channel)
        self.att = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=input_size
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out = self.emb(X)
        out, w = self.att(out, out, out)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
        #return optimizer