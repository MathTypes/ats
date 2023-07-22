import logging
from pathlib import Path

from typing import Type

# find optimal learning rate
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ats.model.eval_callback import WandbClfEvalCallback

torch.manual_seed(0)
np.random.seed(0)

LOG_EVERY_N_STEPS = 10000
BASE_DIR = Path(__file__).parent
LIGHTNING_DIR = BASE_DIR.joinpath("data/lightning")
MODELS_DIR = LIGHTNING_DIR.joinpath("models")


class Time2Vec(nn.Module):
    """General Time2Vec Embedding/Encoding Layer.

    The input data should be related with timestamp/datetime.
        * Input shape: (*, feature_number), where * means any number of
        dimensions.
        * Output shape: (*, linear_channel + period_channel).

    Parameters
    ----------
    linear_channel : int
        The number of linear transformation elements.
    period_channel : int
        The number of cyclical/periodical transformation elements.
    input_channel : int
        The feature number of input data. Default = 1
    period_activation : int
        The activation function for periodical transformation. Default is sine
        function.

    References
    ----------
    .. [1] garyzccisme - Time2Vec
        https://github.com/garyzccisme/Time2Vec/blob/main/time2vec.py

    .. [2] Time2Vec: Learning a Vector Representation of Time
       https://arxiv.org/abs/1907.05321
    """

    def __init__(
        self,
        linear_channel: int,
        period_channel: int,
        input_channel: int = 1,
        period_activation=torch.sin,
    ):
        super().__init__()

        self.linear_channel = linear_channel
        self.period_channel = period_channel
        # logging.info(f"input_channel:{input_channel}, linear_channel:{linear_channel}")
        self.linear_fc = nn.Linear(input_channel, linear_channel)
        self.period_fc = nn.Linear(input_channel, period_channel)
        self.period_activation = period_activation

    def forward(self, x):
        # logging.info(f"x:{x.shape}")
        linear_vec = self.linear_fc(x)
        # logging.info(f"linear_vec:{linear_vec.shape}")
        period_vec = self.period_activation(self.period_fc(x))
        # logging.info(f"period_vec:{period_vec.shape}")
        return torch.cat([linear_vec, period_vec], dim=-1)


class Pipeline:
    """
    Class to ease the running of multiple experiments.
    """

    def __init__(self, config):
        self.model = None
        self.data_module = None
        self.history = None
        self.y_pred = None

        self.preds = None
        self.tests = None
        self.config = config
        self.device = config.job.device

    def create_model(self):
        pass

    def set_learning_rate(self):
        res = Tuner(self.trainer).lr_find(
            self.model,
            train_dataloaders=self.data_module.train_dataloader(),
            val_dataloaders=self.data_module.val_dataloader(),
            early_stop_threshold=None,
            num_iter=self.config.job.num_tune_iter,
            max_lr=0.1,
            min_lr=1e-3,
        )
        suggested_learning_rate = res.suggestion()
        logging.info(f"suggesting learning rate:{res.suggestion()}")
        if not suggested_learning_rate:
            logging.info(f"can not find learning rate!")
        else:
            self.model.hparams.learning_rate = suggested_learning_rate

    def create_trainer(self):
        self.config
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODELS_DIR, monitor="val_loss", save_top_k=1, verbose=True
        )
        # es = EarlyStopping(monitor="val_loss", mode="min", patience=16)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        devices = 1
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project="ATS", log_model=True)
        logging.info(f"data_module:{self.data_module}")
        prediction_logger = WandbClfEvalCallback(
            self.data_module, self.targets, self.config
        )
        # sim_logger = SimilarityLogger()
        self.trainer = pl.Trainer(
            max_epochs=self.config.job.max_epochs,
            logger=wandb_logger,
            callbacks=[
                checkpoint_callback,
                lr_monitor,
                # log_predictions_callback,
                prediction_logger,
                # WandbModelCheckpoint("models"),
                # WandbMetricsLogger(),
                # StochasticWeightAveraging(swa_lrs=1e-2)
            ],
            devices=devices,
            accelerator="gpu",
            # accumulate_grad_batches=8,
            # stochastic_weight_avg=True,
            # precision="bf16",
            # gradient_clip_val=0.5,
            default_root_dir=LIGHTNING_DIR,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            detect_anomaly=True,
            # profiler="advanced",
            # precision='16-mixed',
            # train in half precision
            deterministic=False,
            # check_val_every_n_epoch=10,
            strategy="auto",
        )

    def tune_model(self):
        pass

    def train_model(self):
        logging.info(f"MODELS_DIR:{MODELS_DIR}")
        with torch.cuda.amp.autocast(enabled=False):
            self.history = self.trainer.fit(self.model, self.data_module)
        # evaluate the model on a test set
        # self.trainer.test(datamodule=self.data_module, ckpt_path='best')  # uses last-saved model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch
import torch.nn as nn
from typing import Type
import numpy as np
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"


class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.
        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.
        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.
    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.
    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=1e-8)
        self.mean = mean

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.
        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.
        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    assert x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


def _easy_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: Type[nn.Module],
) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """
    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)


def plot_single_series(samples, target, timesteps, index):
    s_samples = samples[0, index, :, :].detach().cpu().numpy()
    s_timesteps = timesteps[0, :].cpu().numpy()
    s_target = target[0, index, :].cpu().numpy()

    plt.figure()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        plt.fill_between(
            s_timesteps,
            np.quantile(s_samples, quant, axis=1),
            np.quantile(s_samples, 1 - quant, axis=1),
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )
    plt.plot(
        s_timesteps,
        np.quantile(s_samples, 0.5, axis=1),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    plt.plot(
        s_timesteps,
        s_target,
        color=(0, 0, 0),
        linewidth=2,
        zorder=5,
        label="ground truth",
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.show()


def hourly_results(v_plus, bid_price, v_neg, offer_price, da, rt):
    return (v_plus * (da <= bid_price) * (rt - da)) + (
        v_neg * (offer_price < da) * (da - rt)
    )


def worst_loss(results):
    return min(results.sum(axis=1))


def best_loss(results):
    return max(results.sum(axis=1))
