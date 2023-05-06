import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


from data_module import AtsDataModule
#from torch.utils.data import DataLoader, random_split
#from torchfitter.trainer import Trainer
#from torchfitter.utils.data import DataWrapper
#from torchfitter.callbacks import (
#    EarlyStopping,
#    RichProgressBar,
#    LearningRateScheduler
#)
import wandb
from wandb.keras import WandbCallback
from datasets import (
    generate_sine_waves,
    generate_stock_returns,
    generate_white_noise,
    generate_venezia_high_waters,
)

torch.manual_seed(0)
np.random.seed(0)


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
        
        self.linear_fc = nn.Linear(input_channel, linear_channel)
        self.period_fc = nn.Linear(input_channel, period_channel)
        self.period_activation = period_activation

    def forward(self, x):
        linear_vec = self.linear_fc(x)
        period_vec = self.period_activation(self.period_fc(x))
        return torch.cat([linear_vec, period_vec], dim=-1)


class Pipeline:
    """
    Class to ease the running of multiple experiments.
    """
    def __init__(self):
        self.model = None
        #self.X_train = None
        #self.y_train = None
        #self.X_val = None
        #self.y_val = None
        #self.X_test = None
        #self.y_test = None
        self.data_module = None
        self.history = None
        self.y_pred = None

        self.preds = None
        self.tests = None

    def create_model(self):
        pass


    def train_model(self):
        # ---------------------------------------------------------------------
        #criterion = nn.HuberLoss()
        #optimizer = optim.NAdam(self.model.parameters(), lr=0.005)
        #sch = optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, factor=0.7, patience=20, min_lr=0.0001
        #)
        #run = wandb.init(project="WandBAndKerasTuner")
        # ---------------------------------------------------------------------
        callbacks = [
            #EarlyStopping(patience=90, load_best=True),
            #RichProgressBar(display_step=5, log_lr=True),
            #LearningRateScheduler(scheduler=sch, on_train=False),
            #TorchFitterWandbCallback(WandbCallback())
        ]
        wandb_logger = WandbLogger()
        trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
        self.history = trainer.fit(self.model, self.data_module)
        #trainer = Trainer(
        #    model=self.model,
        #    criterion=criterion,
        #    optimizer=optimizer,
        #    callbacks=callbacks,
        #    mixed_precision="no"
        #    #'no', 'fp8', 'fp16', 'bf16'
        #)

        # ---------------------------------------------------------------------
        #history = trainer.fit(train_load, val_loader, epochs=20)
        #self.history = history
        #run.finish()
        # to avoid memory problems
        test_wrapper = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        #test_wrapper = DataWrapper(
        #    self.X_test, self.y_test, dtype_X="float", dtype_y="float"
        #)
        sampler = SequentialSampler(test_wrapper)
        #test_loader = DataLoader(
        #    test_wrapper,
        #    batch_size=64,
        #    pin_memory=True,
        #    sampler=sampler,
        #    shuffle=False
        #)

        #y_pred = trainer.predict(test_loader, as_array=True)
        #y_test = self.y_test

        # only works if predict horizon is 1
        #preds = np.stack([row.flatten() for row in y_pred])
        #tests = np.stack([row.flatten() for row in y_test])

        #self.preds = preds
        #self.tests = tests


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)