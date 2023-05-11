import statistics
import torch
import logging
import torch.nn as nn
from utils import Time2Vec
import wandb
import pytorch_lightning as pl
torch.manual_seed(0)
from torch.nn import functional as F

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
        n_layers=2,
        dropout_rate=0.2
    ):
        super(AttentionEmbeddingLSTM, self).__init__()
        #self.criterion = nn.HuberLoss()
        #self.criterion = torch.nn.MAELoss(reduction='sum')
        self.criterion = torch.nn.L1Loss()
        self.emb = Time2Vec(linear_channel, period_channel, input_channel)
        self.att = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=input_size
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.lin = nn.Linear(hidden_size, out_size)
        self.val_outptus = []
        self.test_outputs = []

    def forward(self, X):
        out = self.emb(X)
        out, w = self.att(out, out, out)
        out, (h, c) = self.lstm(out)
        out = self.dropout(out)
        out = self.lin(out)
        return out

    def compute_loss(self, y_hat, y):
        y_hat[:,0:3,:] = 0
        y[:,0:3,:] = 0
        y_hat[:,4,:] = 0
        y[:,4,:] = 0
        loss = self.criterion(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[:,1:,...]
        y = y[:,1:,...]
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logging.info(f"x:{x.shape}")
        logging.info(f"y:{y.shape}")
        x = x[:,1:,...]
        y = y[:,1:,...]
        y_hat = self.forward(x)
        logging.info(f"y_hat:{y_hat.shape}")
        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x[:,1:,...]
        y = y[:,1:,...]
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)
        self.log('test_loss', loss)

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)
    
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
        #return optimizer


