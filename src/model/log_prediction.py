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
import timeseries_utils
import timeseries_dataset
from torch.utils.data import DataLoader


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, X_test, window_size, step_size, enc_seq_len,
                 dec_seq_len, output_sequence_length, device, num_samples=2048):
        '''method used to define our model parameters'''
        super().__init__()
        self.wandb_logger = wandb_logger
        self.X_test = X_test
        self.criterion = torch.nn.L1Loss(reduction="none").to(device)
        self.test_indices = timeseries_utils.get_indices_entire_sequence(
            data=self.X_test[-1024:].numpy(), 
            window_size=window_size, 
            step_size=step_size)
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.output_sequence_length = output_sequence_length

    def topk_by_sort(self, input, k, axis=None, ascending=True):
        if not ascending:
            input *= -1
        ind = np.argsort(input, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind, axis=axis) 
        return ind, val

    def compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)
        return loss

    def on_validation_epoch_end(self, trainer, pl_module):
        src_vec = []
        tgt_y_vec = []
        time_vec = []
        dev = pl_module.device
        self.val_wrapper = timeseries_dataset.TransformerDataset(
            data=self.X_test[-1024:],
            indices=self.test_indices,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.output_sequence_length
            )
        for i, batch in enumerate(self.val_wrapper):
            src, _, tgt_y = batch
            time = src[:,5]
            src = src[:,:5]
            tgt_y = tgt_y[:,:5]
            src_vec.append(src)
            time_vec.append(time)
            tgt_y_vec.append(tgt_y)
            if i > 256:
                break
            if i % 16 == 0:
                logging.info(f"logging prediction:{i}, {i%16}, len:{len(src_vec)}")
            if len(src_vec) == 64:
                src = torch.from_numpy(np.stack(src_vec))
                tgt_y = torch.from_numpy(np.stack(tgt_y_vec))
                times = torch.from_numpy(np.stack(time_vec)).squeeze(-1)
                if pl_module.batch_first == False:
                    shape_before = src.shape
                    src = src.permute(1, 0, 2)
                    times = times.permute(1, 0)
                    if tgt_y.dim() == 3:
                        tgt_y = tgt_y.permute(1, 0, 2)
                    else:
                        tgt_y = tgt_y.permute(1, 0)
                src = src.to(dev)
                tgt_y = tgt_y[:,:,3].to(dev)
                
                prediction = inference.run_encoder_decoder_inference(
                    model=pl_module.to(dev), 
                    src=src, 
                    forecast_window=pl_module.forecast_window,
                    batch_size=src.shape[1]
                    ).squeeze().to(dev)
                
                loss = self.compute_loss(prediction, tgt_y)
                loss = torch.mean(loss, dim=0).squeeze(0)
                loss = loss.to("cpu").numpy()
                top_ind, top_loss = self.topk_by_sort(loss, 10)
                for ind in top_ind:
                    fig = plt.figure()
                    fig.set_figwidth(40)
                    x = src[:,ind,:].cumsum(dim=1).cpu()
                    pred = prediction[:,ind].cumsum(dim=0).cpu()
                    y = tgt_y[:,ind].cumsum(dim=0).cpu()
                    time = times[:,ind].cpu()
                    close = x[:,3]
                    pred_close = pred[:].squeeze(-1)
                    y_close = y[:].squeeze(-1)
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.plot(np.arange(close.shape[0]), close, label='Training data')
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], close[-1] + pred_close)), label='Prediction', color="red")
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], close[-1] + y_close)), label='Groud Truth', color="purple")
                    now = datetime.fromtimestamp(int(time.numpy()[-1])).strftime("%y-%m-%d %H:%M")
                    ax1.set_xlabel(f'{now}')
                    ax1.set_ylabel('y')
                    self.wandb_logger.log_image(f"top-loss-chart-{now}", images=[fig])
                top_ind, top_loss = self.topk_by_sort(-loss, 10)
                for ind in top_ind:
                    fig = plt.figure()
                    fig.set_figwidth(40)
                    x = src[:,ind,:].cumsum(dim=1).cpu()
                    pred = prediction[:,ind].cumsum(dim=0).cpu()
                    y = tgt_y[:,ind].cumsum(dim=0).cpu()
                    time = times[:,ind].cpu()
                    close = x[:,3]
                    pred_close = pred[:].squeeze(-1)
                    y_close = y[:].squeeze(-1)
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.plot(np.arange(close.shape[0]), close, label='Training data')
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], close[-1] + pred_close)), label='Prediction', color="red")
                    ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred.shape[0]), np.concatenate(([close[-1]], close[-1] + y_close)), label='Groud Truth', color="purple")
                    now = datetime.fromtimestamp(int(time.numpy()[-1])).strftime("%y-%m-%d %H:%M")
                    ax1.set_xlabel(f'{now}')
                    ax1.set_ylabel('y')
                    self.wandb_logger.log_image(f"bottom-loss-chart-{now}", images=[fig])
                src_vec.clear()
                tgt_y_vec.clear()


class LSTMLogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, val_samples, num_samples=2048):
        '''method used to define our model parameters'''
        super().__init__()
        self.wandb_logger = wandb_logger
        self.val_inputs, self.val_labels = val_samples
        self.val_inputs = self.val_inputs[-num_samples:]
        self.val_labels = self.val_labels[-num_samples:]
        #self.val_times = self.val_inputs[:,0,...]
        #self.val_inputs = self.val_inputs[:,1:,...]
        #self.val_labels = self.val_labels[:,1:,...]
        self.criterion = torch.nn.L1Loss(reduction="none")
        #logging.info(f"val_times:{self.val_times}")
        #logging.info(f"val_times_shape:{self.val_times.shape}")
        logging.info(f"val_inputs_shape:{self.val_inputs.shape}")
        logging.info(f"val_labels_shape:{self.val_labels.shape}")

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
        val_inputs = self.val_inputs.to(device=pl_module.device)
        preds = pl_module(val_inputs, return_dict=True)[0]

        #logging.info(f"pred:{preds[0]}")
        #logging.info(f"val_labels:{self.val_labels[0]}")
        metrics = self.criterion(preds, self.val_labels.to(device=pl_module.device)).cpu()
        metrics = torch.sum(metrics, dim=2)
        metrics = torch.sum(metrics, dim=1)
        #logging.info(f"metrics:{metrics.shape}")
        #logging.info(f"metrics:{metrics}")
        ind = np.argsort(metrics, axis=0)
        #logging.info(f"ind:{ind}")
        ind = np.take(ind, np.arange(128), axis=0)
        val_inputs = self.val_inputs[ind]
        preds = preds[ind]
        val_labels = self.val_labels[ind]
        #val_times = self.val_times[ind]
        #logging.info(f"after ind:{ind}")

        #logging.info(f"preds:{preds.shape}")
        #logging.info(f"val_inputs:{val_inputs.shape}")
        #logging.info(f"val_labels:{val_labels.shape}")
        #fig = plt.figure(figsize=(400,200))
        fig_cnt = val_inputs.shape[0] // 4
        for i in range(0, fig_cnt):
            fig = plt.figure()
            fig.set_figwidth(40)
            for j in range(0, 4):
                x = val_inputs[i*4+j].cpu()
                pred = preds[i*4+j].cpu()
                y = val_labels[i*4+j].cpu()
                #times = val_times[i*4+j].cpu()
                open = x[0]
                high = x[1]
                low = x[2]
                close = x[3]
                pred_close = pred[3]
                #logging.info(f"pred_close:{pred_close.shape}")
                #pred_close = pred
                y_close = y[3]
                #y_close = y
                #logging.info(f"time:{times}")
                ax1 = fig.add_subplot(1, 4, j+1)
                ax1.plot(np.arange(close.shape[0]), close, label='Training data')
                ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred_close.shape[0]), np.concatenate(([close[-1]], pred_close)), label='Prediction', color="red")
                ax1.plot(np.arange(close.shape[0]-1, close.shape[0]+pred_close.shape[0]), np.concatenate(([close[-1]], y_close)), label='Groud Truth', color="purple")
                #now = datetime.fromtimestamp(times.numpy()[-1])
                #ax1.set_xlabel(f'{now.strftime("%y-%m-%d %H:%M")}')
                ax1.set_ylabel('y')
            self.wandb_logger.log_image(f"chart-{i}", images=[fig])
