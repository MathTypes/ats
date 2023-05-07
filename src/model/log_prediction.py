import io
from PIL import Image
import logging
from pytorch_lightning.callbacks import Callback
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, val_samples, num_samples=16):
        '''method used to define our model parameters'''
        super().__init__()
        self.wandb_logger = wandb_logger
        self.val_inputs, self.val_labels = val_samples
        self.val_inputs = self.val_inputs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        self.val_times = self.val_inputs[:,0,...]
        self.val_inputs = self.val_inputs[:,1:,...]
        self.val_labels = self.val_labels[:,1:,...]
        logging.info(f"val_times:{self.val_times}")
        logging.info(f"val_times_shape:{self.val_times.shape}")
        logging.info(f"val_inputs_shape:{self.val_inputs.shape}")
        logging.info(f"val_labels_shape:{self.val_labels.shape}")

    def on_validation_epoch_end(self, trainer, pl_module):
        #wandb.init()
        val_inputs = self.val_inputs.to(device=pl_module.device)
        preds = pl_module(val_inputs)
        logging.info(f"preds:{preds.shape}")
        logging.info(f"val_inputs:{val_inputs.shape}")
        logging.info(f"val_labels:{self.val_labels.shape}")
        #fig = plt.figure(figsize=(400,200))
        fig = plt.figure()
        
        for idx, (x, pred, y) in enumerate(zip(val_inputs, preds, self.val_labels)):
            x = x.cpu()
            pred = pred.cpu()
            y = y.cpu()
            open = x[0]
            high = x[1]
            low = x[2]
            close = x[3]
            pred_close = pred[3]
            #pred_close = pred
            y_close = y[3]
            #y_close = y
            ax1 = fig.add_subplot(4, 4, idx+1)
            ax1.plot(np.arange(close.shape[0]), close, label='Training data')
            ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+5), np.concatenate(([close[-1]], pred_close)), label='Prediction', color="red")
            ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+5), np.concatenate(([close[-1]], y_close)), label='Groud Truth', color="purple")
            ax1.set_xlabel('time period')
            ax1.set_ylabel('y')
        self.wandb_logger.log_image(f"chart-{idx}", images=[fig])
    
        #trainer.logger.experiment.log({
        #    "examples": [wandb.Image(self.generate_image(x, pred, y), caption=f"Pred:{pred}, Label:{y}") 
        #                    for x, pred, y in zip(val_inputs, preds, self.val_labels)],
        #    "global_step": trainer.global_step
        #    })
        
        #captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(self.val_labels, preds)]
        
        # Option 1: log images with `WandbLogger.log_image`
        #self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)
        # Option 2: log predictions as a Table
        #columns = ['image', 'ground truth', 'prediction']
        #data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
        columns = ['ground truth', 'prediction']
        data = [[y_i, y_pred] for x_i, y_i, y_pred in list(zip(self.val_inputs, self.val_labels, preds))]
        self.wandb_logger.log_table(key='sample_table', columns=columns, data=data)

