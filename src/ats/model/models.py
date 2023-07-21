from typing import Optional, Tuple, Union
import torch
import logging
import torch.nn as nn
import wandb
import pytorch_lightning as pl

torch.manual_seed(0)
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    SampleTSPredictionOutput,
    Seq2SeqTSModelOutput,
    Seq2SeqTSPredictionOutput,
)
from transformers.time_series_utils import (
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from transformers.models.time_series_transformer.configuration_time_series_transformer import (
    TimeSeriesTransformerConfig,
)

from ats.market_data import timeseries_utils as ts_utils
from ats.model.utils import Time2Vec

class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


def weighted_average(
    input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, input_tensor * weights, torch.zeros_like(input_tensor)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def create_mlp(num_inputs, hidden_size, num_outputs, seq_len):
    # MLP as a CNN
    mlp = nn.Sequential(
        nn.Conv1d(num_inputs, hidden_size, seq_len),
        nn.Tanh(),
        nn.Conv1d(hidden_size, hidden_size, seq_len),
        nn.Tanh(),
        nn.Conv1d(hidden_size, num_outputs, seq_len),
        nn.LogSoftmax(dim=seq_len),
    )
    return mlp


class AttentionEmbeddingLSTM(pl.LightningModule):
    """
    Time2vec embedding + Attention + LSTM.
    """

    def __init__(
        self,
        input_features,
        linear_channel,
        period_channel,
        input_size,
        out_size,
        out_values,
        hidden_size=4,
        n_layers=2,
        dropout_rate=0.2,
        config: TimeSeriesTransformerConfig = None,
    ):
        super(AttentionEmbeddingLSTM, self).__init__()
        self.scaler = ts_utils.TimeSeriesStdScaler(dim=1, keepdim=True)
        # self.criterion = nn.HuberLoss()
        # self.criterion = torch.nn.MAELoss(reduction='sum')
        self.out_values = out_values
        self.criterion = nll
        # Build a one-dimensional convolutional neural layer
        # See https://stats.stackexchange.com/questions/380996/convolutional-network-how-to-choose-output-channels-number-stride-and-padding/381032#381032
        # for how output_channel is computed.
        # output_channel = ((n-f+2p)/s)+1
        # where n is the pixels of the image i.e. input_size
        # f is the number of kernels, in our case it is 3*3 kernel which mean f = 3
        # p is the padding, p = 0
        # s is the stride, s = 0
        kernel_size = 4
        stride = 2
        output_channels = int(((input_size - kernel_size + 2 * 0) / stride) + 1)
        logging.info(
            f"output_channels:{output_channels}, input_features:{input_features}"
        )
        self.cov1d = nn.Conv1d(
            in_channels=input_features,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
        )
        # self.emb = Time2Vec(linear_channel, period_channel, input_size)
        logging.info(
            f"linear_channel:{linear_channel}, period_channel:{period_channel}, output_channels:{output_channels}"
        )
        self.emb = Time2Vec(
            linear_channel, input_size - linear_channel, output_channels
        )
        self.att = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=input_size
            # embed_dim=output_channels, num_heads=output_channels
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            # input_size=output_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.distribution_output = NormalOutput(dim=out_size)
        self.parameter_projection = self.distribution_output.get_parameter_projection(1)
        self.target_shape = self.distribution_output.event_shape
        # logging.info(f"hidden_size:{hidden_size}")
        logging.info(f"target_shape:{self.target_shape}")
        self.lin = nn.Linear(hidden_size, self.out_values)
        self.relu = nn.ReLU()
        self.val_outptus = []
        self.test_outputs = []

    def output_params(self, dec_output):
        return self.parameter_projection(dec_output[:, -1:])

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        # logging.info(f"sliced_params:{type(sliced_params)}")
        return self.distribution_output.distribution(
            torch.stack([sliced_params[0], sliced_params[1]], dim=-1),
            loc=loc,
            scale=scale,
        )

    def forward(
        self,
        X,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqTSModelOutput, Tuple]:
        past_values = X
        past_observed_mask = torch.ones_like(past_values)
        _, loc, scale = self.scaler(past_values, past_observed_mask)
        logging.info(f"loc:{loc.shape}")
        logging.info(f"scale:{scale.shape}")
        # logging.info(f"X:{X.shape}")
        # logging.info(f"future_values:{future_values.shape}")
        # logging.info(f"loc:{loc.shape}")
        # logging.info(f"before scale:{past_values}")
        inputs = (past_values - loc) / scale
        # logging.info(f"after scale:{inputs}")
        logging.info(f"inputs_after_scale:{inputs.shape}")
        inputs = self.cov1d(inputs).squeeze()
        logging.info(f"inputs_after_cov1d:{inputs.shape}")
        out = self.emb(inputs)
        logging.info(f"out_after_emb:{out.shape}")
        out, w = self.att(out, out, out)
        out, (h, c) = self.lstm(out)
        logging.info(f"out_after_attention:{out.shape}")
        hidden_state = self.dropout(out)
        logging.info(f"hidden_state1:{hidden_state.shape}")
        hidden_state = self.relu(hidden_state)
        logging.info(f"hidden_state_sigmoid:{hidden_state.shape}")
        hidden_state = self.lin(hidden_state)
        logging.info(f"hidden_state_lin:{hidden_state.shape}")
        params = self.output_params(hidden_state)  # outputs.last_hidden_state
        logging.info(f"params0:{params[0]}")
        logging.info(f"params1:{params[1]}")
        distribution = self.output_distribution(
            params, loc=loc, scale=scale, trailing_n=2
        )
        prediction_loss = None
        logging.info(f"distribution:{distribution}")
        if future_values is not None:
            # logging.info(f"future_values:{future_values}")
            logging.info(f"before future_values:{future_values.shape}")
            future_values = (future_values - loc[:, 3, :]) / scale[:, 3, :]
            logging.info(f"scaled future_values:{future_values.shape}")
            loss = self.criterion(distribution, future_values)

            future_observed_mask = None
            if future_observed_mask is None:
                future_observed_mask = torch.ones_like(future_values)

            if len(self.target_shape) == 0:
                loss_weights = future_observed_mask
            else:
                loss_weights, _ = future_observed_mask.min(dim=-1, keepdim=False)

            prediction_loss = weighted_average(loss, weights=loss_weights)

        return Seq2SeqTSPredictionOutput(
            params=params,
            loc=loc,
            scale=scale,
            loss=prediction_loss,
        )
        return out

    def compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, return_dict=True, future_values=y)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, return_dict=True, future_values=y)
        loss = output.loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, return_dict=True, future_values=y)
        loss = output.loss
        self.log("test_loss", loss)

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
        # return optimizer
