import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.callbacks import GradientAccumulationScheduler, StochasticWeightAveraging
#import pytorch_lightning as pl
import lightning.pytorch as pl
#from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import WandbLogger
#from pytorch_lightning import Trainer

from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from eval_callback import WandbClfEvalCallback

from data_module import LSTMDataModule, TransformerDataModule
from log_prediction import LogPredictionsCallback, LSTMLogPredictionsCallback, TSLogPredictionsCallback
import wandb
from pathlib import Path
#from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import data_module
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
        period_activation=torch.sin
    ):
        super().__init__()
        
        self.linear_channel = linear_channel
        self.period_channel = period_channel
        #logging.info(f"input_channel:{input_channel}, linear_channel:{linear_channel}")
        self.linear_fc = nn.Linear(input_channel, linear_channel)
        self.period_fc = nn.Linear(input_channel, period_channel)
        self.period_activation = period_activation

    def forward(self, x):
        #logging.info(f"x:{x.shape}")
        linear_vec = self.linear_fc(x)
        #logging.info(f"linear_vec:{linear_vec.shape}")
        period_vec = self.period_activation(self.period_fc(x))
        #logging.info(f"period_vec:{period_vec.shape}")
        return torch.cat([linear_vec, period_vec], dim=-1)


class Pipeline:
    """
    Class to ease the running of multiple experiments.
    """
    def __init__(self, device):
        self.model = None
        self.data_module = None
        self.history = None
        self.y_pred = None

        self.preds = None
        self.tests = None
        self.device = device

    def create_model(self):
        pass


    def train_model(self):
        logging.info(f"MODELS_DIR:{MODELS_DIR}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODELS_DIR,
            monitor="val_loss",
            save_top_k=1,
            verbose=True
        )
        #es = EarlyStopping(monitor="val_loss", mode="min", patience=16)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        wandb_logger = WandbLogger(project='ATS', log_model=True)
        #log_predictions_callback = LogPredictionsCallback(wandb_logger,
        #                                                  self.data_module.X_test,
        #                                                  window_size=self.data_module.window_size,
        #                                                  step_size=self.data_module.step_size,
        #                                                  enc_seq_len=self.data_module.enc_seq_len,
        #                                                  dec_seq_len=self.data_module.dec_seq_len,
        #                                                  device="cuda",
        #                                                  output_sequence_length=self.data_module.output_sequence_length)
        #log_predictions_callback = TSLogPredictionsCallback(wandb_logger, self.data_module.test_dataloader())
        # till 5th epoch, it will accumulate every 8 batches. From 5th epoch
        # till 9th epoch it will accumulate every 4 batches and after that no accumulation
        # will happen. Note that you need to use zero-indexed epoch keys here
        #accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
        #devices = 1 if self.device == "cpu" else -1
        devices = 1
        logging.info(f"device:{self.device}")
        #profiler = AdvancedProfiler()
        wandb_logger = WandbLogger(project='ATS', log_model=True)
        logging.info(f"data_module:{self.data_module}")
        prediction_logger = WandbClfEvalCallback(
            self.data_module.val_dataloader(),
            data_table_columns=["time", "close_pct"],
            pred_table_columns=["epoch", "time", "close_pct", "pred_close_pct"])
        trainer = pl.Trainer(max_epochs=100, logger=wandb_logger,
                             callbacks=[checkpoint_callback, lr_monitor,
                                        #log_predictions_callback,
                                        prediction_logger,
                                        #WandbModelCheckpoint("models"),
                                        #WandbMetricsLogger(),
                                        #StochasticWeightAveraging(swa_lrs=1e-2)
                             ],
                             devices=devices,
                             #accelerator=self.device,
                             accelerator="gpu",
                             accumulate_grad_batches=8,
                             #stochastic_weight_avg=True,
                             #precision="bf16",
                             gradient_clip_val=0.5,
                             default_root_dir=LIGHTNING_DIR,
                             log_every_n_steps=LOG_EVERY_N_STEPS,
                             detect_anomaly=True,
                             precision=16,
                             #profiler="advanced",
                             #precision='16-mixed',
                             # train in half precision
                             deterministic=False, strategy='auto')
        self.history = trainer.fit(self.model, self.data_module)
        # evaluate the model on a test set
        trainer.test(datamodule=self.data_module, ckpt_path='best')  # uses last-saved model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
