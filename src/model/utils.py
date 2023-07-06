import datetime
import logging
from pathlib import Path

# find optimal learning rate
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import data_module
from data_module import LSTMDataModule, TransformerDataModule
from eval_callback import WandbClfEvalCallback
from log_prediction import LogPredictionsCallback, LSTMLogPredictionsCallback, TSLogPredictionsCallback

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
    def __init__(self, config):
        self.model = None
        self.data_module = None
        self.history = None
        self.y_pred = None

        self.preds = None
        self.tests = None
        self.config = config
        self.device = config.job.device
        self.train_start_date = datetime.datetime.strptime(config.job.train_start_date,"%Y-%m-%d")
        self.test_start_date = datetime.datetime.strptime(config.job.test_start_date,"%Y-%m-%d")
        self.test_end_date = datetime.datetime.strptime(config.job.test_end_date,"%Y-%m-%d")

    def create_model(self):
        pass

    def set_learning_rate(self):
        res = Tuner(self.trainer).lr_find(self.model,
                                          train_dataloaders=self.data_module.train_dataloader(),
                                          val_dataloaders=self.data_module.val_dataloader(),
                                          early_stop_threshold=None,
                                          max_lr=0.1,
                                          min_lr=1e-3)
        suggested_learning_rate = res.suggestion()
        logging.info(f"suggesting learning rate:{res.suggestion()}")
        if not suggested_learning_rate:
            logging.info(f"can not find learning rate!")
            exit(0)
        self.model.hparams.learning_rate = suggested_learning_rate


    def create_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODELS_DIR,
            monitor="val_loss",
            save_top_k=1,
            verbose=True
        )
        #es = EarlyStopping(monitor="val_loss", mode="min", patience=16)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        devices = 1
        logging.info(f"device:{self.device}")
        wandb_logger = WandbLogger(project='ATS', log_model=True)
        logging.info(f"data_module:{self.data_module}")
        prediction_logger = WandbClfEvalCallback(self.data_module, self.targets, self.config)
        #sim_logger = SimilarityLogger() 
        self.trainer = pl.Trainer(max_epochs=self.config.job.max_epochs, logger=wandb_logger,
                             callbacks=[checkpoint_callback, lr_monitor,
                                        #log_predictions_callback,
                                        prediction_logger,
                                        #WandbModelCheckpoint("models"),
                                        #WandbMetricsLogger(),
                                        #StochasticWeightAveraging(swa_lrs=1e-2)
                             ],
                             devices=devices,
                             accelerator="gpu",
                             accumulate_grad_batches=8,
                             #stochastic_weight_avg=True,
                             #precision="bf16",
                             gradient_clip_val=0.5,
                             default_root_dir=LIGHTNING_DIR,
                             log_every_n_steps=LOG_EVERY_N_STEPS,
                             detect_anomaly=True,
                             #profiler="advanced",
                             #precision='16-mixed',
                             # train in half precision
                             deterministic=False,
                             #check_val_every_n_epoch=10,
                             strategy='auto',)
        
    def tune_model(self):
        pass

    
    def train_model(self):
        logging.info(f"MODELS_DIR:{MODELS_DIR}")
        self.history = self.trainer.fit(self.model, self.data_module)
        # evaluate the model on a test set
        #self.trainer.test(datamodule=self.data_module, ckpt_path='best')  # uses last-saved model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
