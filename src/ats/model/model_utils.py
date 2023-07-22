# Usage
import logging
from math import ceil
import os
import time
from typing import List
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import lightning.pytorch as pl
from omegaconf import OmegaConf
from pytorch_forecasting import (
    TemporalFusionTransformer,
    PatchTstTransformer,
    PatchTstTftTransformer,
    PatchTftSupervised,
)
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    MAPCSE,
    RMSE,
    SMAPE,
    QuantileLoss,
    MQF2DistributionLoss,
    MultiLoss,
    SharpeLoss,
    DistributionLoss,
)
from pytorch_forecasting import NHiTS
from torch import nn

from ats.market_data import market_data_mgr
from ats.market_data.data_module import TimeSeriesDataModule
from ats.model.loss import MultiLossWithUncertaintyWeight
from ats.model import nhits_tuner
from ats.util import time_util


def create_loss(loss_name, device, prediction_length=None, hidden_size=None):
    loss = None
    logging.info(f"loss_name:{loss_name}")
    if loss_name == "MASE":
        loss = MASE()
    if loss_name == "SharpeLoss":
        loss = SharpeLoss()
    if loss_name == "SMAPE":
        loss = SMAPE()
    if loss_name == "MAE":
        loss = MAE()
    if loss_name == "RMSE":
        loss = RMSE()
    if loss_name == "MAPE":
        loss = MAPE()
    if loss_name == "MAPCSE":
        loss = MAPCSE()
    if loss_name == "MQF2DistributionLoss":
        loss = MQF2DistributionLoss(prediction_length=prediction_length)
    if loss_name == "QuantileLoss":
        loss = QuantileLoss()
    loss = loss.to(device)
    return loss


def get_logging_metrics(logging_metrics, device, prediction_length):
    metrics = []
    for loss_name in logging_metrics:
        metrics.append(create_loss(loss_name, device, prediction_length))
    return nn.ModuleList(metrics)


# Different heads must still have same prediction_length. Otherwise, the data loading
# would not work since it can only have one sampling mode across heads. It is fine for
# different heads to have different logging_metrics as the actual outputs can be different
# across heads.
def create_loss_per_head(heads, device, prediction_length):
    loss_per_head = {}
    for name, value in heads.items():
        loss = create_loss(value.loss_name, device, prediction_length)
        logging_metrics = get_logging_metrics(
            value.logging_metrics, device, prediction_length
        )
        loss_per_head[name] = {"loss": loss, "logging_metrics": logging_metrics}
    return loss_per_head


def get_loss(config, prediction_length=None, hidden_size=None):
    target_size = 1
    if not prediction_length:
        prediction_length = config.model.prediction_length
    # logging.info(f"prediction_length:{prediction_length}")
    if OmegaConf.is_list(config.model.target):
        target = OmegaConf.to_object(config.model.target)
        logging.info(f"target:{target}")
        target_size = len(target)
        loss_name = OmegaConf.to_object(config.model.loss_name)
        losses = []
        for i in range(target_size):
            losses.append(
                create_loss(loss_name[i], config.job.device, prediction_length)
            )
        loss = MultiLoss(losses)
    else:
        loss_name = config.model.loss_name
        loss = create_loss(loss_name, config.job.device, prediction_length)
    logging.info(f"created loss:{loss}")
    return loss


def get_nhits_model(config, data_module, loss):
    config.job.device
    training = data_module.training
    config.model.prediction_length
    # configure network and trainer
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    net = NHiTS.from_dataset(
        training,
        weight_decay=1e-2,
        loss=loss,
        backcast_loss_ratio=0.0,
        hidden_size=config.model.hidden_size,
        prediction_length=config.model.prediction_length,
        context_length=config.model.context_length,
        learning_rate=config.model.learning_rate,
        optimizer="AdamW",
        log_interval=0.25,
        # n_blocks=[1,1,1],
        # downsample_frequencies=[1,23,46],
        # n_layers=2,
        # log_val_interval=10000
    )
    return net


def get_tft_model(config, data_module):
    config.job.device
    training = data_module.training
    config.model.prediction_length
    # configure network and trainer
    pl.seed_everything(42)
    loss = get_loss(config)
    net = TemporalFusionTransformer.from_dataset(
        training,
        weight_decay=1e-2,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=8,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        loss=loss,
        optimizer="Ranger",
        log_interval=0.25,
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=10,
    )
    return net


def get_patch_tst_model(config, data_module):
    config.job.device
    training = data_module.training
    config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    # configure network and trainer
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pl.seed_everything(42)
    loss = get_loss(config)
    num_patch = (max(context_length, patch_len) - patch_len) // stride + 1
    logging.getLogger().info(
        f"context_length:{context_length}, patch_len:{patch_len}, stride:{stride}, num_patch:{num_patch}"
    )
    net = PatchTstTransformer.from_dataset(
        training,
        patch_len=config.model.patch_len,
        stride=stride,
        num_patch=num_patch,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        d_model=8,  # most important hyperparameter apart from learning rate
        hidden_size=8,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=1,
        loss=loss,
        attn_dropout=0.1,  # between 0.1 and 0.3 are good values
        # hidden_continuous_size=8,  # set to <= hidden_size
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net


def get_patch_tst_tft_model(config, data_module):
    config.job.device
    training = data_module.training
    prediction_length = config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    pl.seed_everything(42)
    d_model = config.model.d_model
    logging.info(f"prediction_length:{prediction_length}, patch_len:{patch_len}")
    loss = get_loss(config, hidden_size=d_model)
    num_patch = (max(context_length, patch_len) - patch_len) // stride + 1
    logging.info(
        f"patch_len:{patch_len}, num_patch:{num_patch}, context_length:{context_length}, stride:{stride}"
    )
    net = PatchTstTftTransformer.from_dataset(
        training,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        loss=loss,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        d_model=d_model,  # most important hyperparameter apart from learning rate
        hidden_size=8,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=1,
        attn_dropout=0.1,  # between 0.1 and 0.3 are good values
        # hidden_continuous_size=8,  # set to <= hidden_size
        # loss=QuantileLoss(),
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net


def get_output_size(loss):
    # logging.error(f"loss:{loss}")
    if isinstance(loss, QuantileLoss):
        # logging.info(f"QuantileLoss:{len(loss.quantiles)}")
        return len(loss.quantiles)
    # elif isinstance(normalizer, NaNLabelEncoder):
    #    #logging.info(f"normalizer.classes_:{len(normalizer.classes_)}")
    #    return len(normalizer.classes_)
    elif isinstance(loss, DistributionLoss):
        return len(loss.distribution_arguments)
    else:
        return 1  # default to 1


def get_patch_tft_supervised_model(config, data_module, heads):
    device = config.job.device
    training = data_module.training
    prediction_length = config.model.prediction_length
    context_length = config.model.context_length
    patch_len = config.model.patch_len
    stride = config.model.stride
    d_model = config.model.d_model
    pl.seed_everything(42)
    logging.info(f"prediction_length:{prediction_length}, patch_len:{patch_len}")
    num_patch = (max(context_length, patch_len) - patch_len) // stride + 1
    prediction_num_patch = (max(prediction_length, patch_len) - patch_len) // stride + 1
    logging.info(
        f"patch_len:{patch_len}, num_patch:{num_patch}, context_length:{context_length}, stride:{stride}, prediction_num_patch:{prediction_num_patch}"
    )
    loss = None
    logging_metrics = []
    output_size_dict = {}
    if config.model.multitask:
        loss_per_head = create_loss_per_head(heads, device, prediction_length)
        losses = []
        for name, l in loss_per_head.items():
            losses.append(l["loss"])
            logging_metrics.append(l["logging_metrics"])
            # logging.info(f"loss name: {name}, l:{l}")
            output_size_dict[name] = get_output_size(l["loss"])
        # logging.info(f"losses:{losses}")
        # logging.info(f"logging_metrics:{logging_metrics}")
        # logging.info(f"output_size_dict:{output_size_dict}")
        if len(losses) > 1:
            loss = MultiLossWithUncertaintyWeight(losses)
        else:
            loss = losses[0]
    else:
        # TODO implement single task loss
        pass

    net = PatchTftSupervised.from_dataset(
        training,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        prediction_num_patch=prediction_num_patch,
        loss=loss,
        loss_per_head=loss_per_head,
        # logging_metrics=logging_metrics,
        logging_metrics=None,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=config.model.learning_rate,
        d_model=d_model,  # most important hyperparameter apart from learning rate
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        lstm_layers=config.model.lstm_layers,
        # number of attention heads. Set to up to 4 for large datasets
        n_heads=config.model.attn_heads,
        attn_dropout=config.model.attn_dropout,  # between 0.1 and 0.3 are good values
        # hidden_continuous_size=8,  # set to <= hidden_size
        # loss=QuantileLoss(),
        optimizer="Ranger",
        output_size=output_size_dict,
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    return net


def run_tune(study_name, config):
    study = nhits_tuner.optimize_hyperparameters(study_name, config)
    print(f"study:{study}")


def get_input_dirs(config):
    base_dir = "data/FUT/30min_rsp/ES"
    for cur_date in time_util.monthlist(config["start_date"], config["end_date"]):
        for_date = cur_date[0]
        date_dir = os.path.join(base_dir, for_date.strftime("%Y%m%d"))
        files = os.listdir(date_dir)
        files = [
            date_dir + "/" + f for f in files if os.path.isfile(date_dir + "/" + f)
        ]  # Filtering only the files.
        input_dirs.extend(files)
    return input_dirs


def get_heads_and_targets(config):
    heads = config.model.heads
    # logging.info(f"heads:{heads}")
    head_dict = {}
    targets = set()
    for head in heads:
        head_dict[head] = config.model[head]
        if isinstance(head_dict[head], List):
            for target in head_dict[head].target:
                targets.add(target)
        else:
            targets.add(head_dict[head].target)
    if len(targets) == 1:
        targets = next(iter(targets))
    else:
        # Need sorted to have consistent target orders. This affects prediction
        # output position from model.
        targets = list(sorted(targets))
    # logging.info(f"head_dict:{head_dict}, targets:{targets}")
    return head_dict, targets


def run_train(config, net, trainer, data_module):
    device = config["device"]
    # fit network
    trainer.fit(
        net,
        train_dataloaders=data_module.train_dataloader,
        val_dataloaders=data_module.val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    NHiTS.load_from_checkpoint(best_model_path).to(device)
    logging.info(f"best_model_path:{best_model_path}")


def week_of_month(dt):
    """Returns the week of the month for the specified date."""
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom / 7.0))
