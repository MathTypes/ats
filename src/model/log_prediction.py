import io
from PIL import Image
from datetime import datetime
import logging
from pytorch_lightning.callbacks import Callback
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import PIL
import inference
PIL.Image.MAX_IMAGE_PIXELS = None

class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, val_dataloader, num_samples=2048):
        '''method used to define our model parameters'''
        super().__init__()
        self.wandb_logger = wandb_logger
        self.val_dataloader = val_dataloader
        self.criterion = torch.nn.L1Loss(reduction="none")

    def topk_by_sort(input, k, axis=None, ascending=True):
        if not ascending:
            input *= -1
        ind = np.argsort(input, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind, axis=axis) 
        return ind, val

    def on_validation_epoch_end(self, trainer, pl_module):
        #wandb.init()
        for i, (src, _, tgt_y) in enumerate(self.val_dataloader):
            prediction = inference.run_encoder_decoder_inference(
                model=pl_module, 
                src=src, 
                forecast_window=pl_module.forecast_window,
                batch_size=src.shape[1]
                )
            logging.info(f"prediction:{prediction}")
            logging.info(f"prediction_shape:{prediction.shape}")
            logging.info(f"tgt_y:{tgt_y}")
            logging.info(f"tgt_y:{tgt_y.shape}")
            metrics = pl_module.criterion(tgt_y, prediction)
            metrics = torch.sum(metrics)
            logging.info(f"metrics:{metrics}")
