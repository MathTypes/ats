"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import copy
import logging
import os
from typing import Any, Dict, Tuple, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
import optuna.logging
import statsmodels.api as sm
import wandb
from pytorch_forecasting.metrics import QuantileLoss

from ats.model import model_utils

optuna_logger = logging.getLogger("optuna")


# need to inherit from callback for this to work
class PyTorchLightningPruningCallbackAdjusted(
    pl.Callback, PyTorchLightningPruningCallback
):
    pass


def optimize_hyperparameters(
    study_name: str,
    config,
    timeout: float = 3600 * 8.0,  # 8 hours
    gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
    context_length_ratio_range: Tuple[int, int] = (3, 3),
    prediction_length_range: Tuple[int, int] = (36, 36),
    static_hidden_size_range: Tuple[int, int] = (8, 64),
    hidden_size_range: Tuple[int, int] = (4, 64),
    dropout_range: Tuple[float, float] = (0.1, 0.5),
    learning_rate_range: Tuple[float, float] = (1e-5, 0.1),
    use_learning_rate_finder: bool = True,
    trainer_kwargs: Dict[str, Any] = {},
    log_dir: str = "lightning_logs",
    study: optuna.Study = None,
    verbose: Union[int, bool] = None,
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
    **kwargs,
) -> optuna.Study:
    """
    Optimize Temporal Fusion Transformer hyperparameters.

    Run hyperparameter optimization. Learning rate for is determined with
    the PyTorch Lightning learning rate finder.

    Args:
        train_dataloaders (DataLoader): dataloader for training model
        val_dataloaders (DataLoader): dataloader for validating model
        model_path (str): folder to which model checkpoints are saved
        max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
        n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
        timeout (float, optional): Time in seconds after which training is stopped regardless of number of epochs
            or validation metric. Defaults to 3600*8.0.
        hidden_size_range (Tuple[int, int], optional): Minimum and maximum of ``hidden_size`` hyperparameter. Defaults
            to (16, 265).
        hidden_continuous_size_range (Tuple[int, int], optional):  Minimum and maximum of ``hidden_continuous_size``
            hyperparameter. Defaults to (8, 64).
        attention_head_size_range (Tuple[int, int], optional):  Minimum and maximum of ``attention_head_size``
            hyperparameter. Defaults to (1, 4).
        dropout_range (Tuple[float, float], optional):  Minimum and maximum of ``dropout`` hyperparameter. Defaults to
            (0.1, 0.3).
        learning_rate_range (Tuple[float, float], optional): Learning rate range. Defaults to (1e-5, 1.0).
        use_learning_rate_finder (bool): If to use learning rate finder or optimize as part of hyperparameters.
            Defaults to True.
        trainer_kwargs (Dict[str, Any], optional): Additional arguments to the
            `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ such
            as ``limit_train_batches``. Defaults to {}.
        log_dir (str, optional): Folder into which to log results for tensorboard. Defaults to "lightning_logs".
        study (optuna.Study, optional): study to resume. Will create new study by default.
        verbose (Union[int, bool]): level of verbosity.
            * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
            * 0 or False: log only warnings.
            * 1 or True: log pruning events.
            * 2: optuna logging level at debug level.
            Defaults to None.
        pruner (optuna.pruners.BasePruner, optional): The optuna pruner to use.
            Defaults to optuna.pruners.SuccessiveHalvingPruner().

        **kwargs: Additional arguments for the :py:class:`~TemporalFusionTransformer`.

    Returns:
        optuna.Study: optuna study results
    """
    model_path = config.model.checkpoint_output_dir
    max_epochs = config.job.max_epochs
    n_trials = config.job.num_tune_iter
    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    loss = kwargs.get(
        "loss", QuantileLoss()
    )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next

    # create objective function
    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(model_path, "trial_{}".format(trial.number)),
            filename="{epoch}",
            monitor="val_loss",
        )

        LearningRateMonitor()
        kwargs["loss"] = copy.deepcopy(loss)
        trial_config = config
        trial_config.update(dict(trial.params))
        trial_config["trial.number"] = trial.number
        # trial_config["dropout"]=trial.suggest_uniform("dropout", *dropout_range)
        # trial_config["hidden_size"] = trial.suggest_int("hidden_size", *hidden_size_range, log=False)
        trial_config["log_mode"] = True
        logging.info(f"trial_config:{trial_config}")
        # trial_config["prediction_length"] = trial.suggest_int("prediction_length", *prediction_length_range, log=True)
        context_length_ratio = trial.suggest_int(
            "context_length_ratio", *context_length_ratio_range, log=True
        )
        trial_config["context_length"] = (
            trial_config["prediction_length"] * context_length_ratio
        )
        # trial_config["loss_name"] = trial.suggest_categorical("loss", ["MASE", "SMAPE", "MAE", "RMSE", "MAPE", "MQF2DistributionLoss"])
        trial_config["min_encoder_length"] = trial_config["context_length"]
        trial_config["max_encoder_length"] = trial_config["context_length"]
        trial_config["min_prediction_length"] = trial_config["prediction_length"]
        trial_config["max_prediction_length"] = trial_config["prediction_length"]
        # gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
        # trial_config["gradient_clip_val"] = gradient_clip_val
        data_module = model_utils.get_data_module(trial_config)
        model = model_utils.get_model(trial_config, data_module)
        # find good learning rate
        if use_learning_rate_finder:
            lr_trainer = model_utils.get_trainer(trial_config, data_module)
            tuner = Tuner(lr_trainer)
            res = tuner.lr_find(
                model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader(),
                early_stop_threshold=10000,
                min_lr=learning_rate_range[0],
                num_training=100,
                max_lr=learning_rate_range[1],
            )

            loss_finite = np.isfinite(res.results["loss"])
            if (
                loss_finite.sum() > 3
            ):  # at least 3 valid values required for learning rate finder
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[min(loss_finite.sum() - 3, 10) : -1].T
                optimal_idx = np.gradient(loss_smoothed).argmin()
                optimal_lr = lr_smoothed[optimal_idx]
            else:
                optimal_idx = np.asarray(res.results["loss"]).argmin()
                optimal_lr = res.results["lr"][optimal_idx]
            optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
            # add learning rate artificially
            model.hparams.learning_rate = trial.suggest_uniform(
                "learning_rate", optimal_lr, optimal_lr
            )
        else:
            model.hparams.learning_rate = trial.suggest_loguniform(
                "learning_rate", *learning_rate_range
            )
        trial_config["learning_rate"] = model.hparams.learning_rate
        # logging.info(f"trial_config_before_model:{trial_config}")
        # trial_config.update(dict(model.hparams))
        wandb.init(
            project="ats-optuna",
            entity="johnnychen7622",
            config=trial_config,
            group=study_name,
            reinit=True,
        )
        # fit
        # trainer = pl.Trainer(
        #    **default_trainer_kwargs,
        # )
        logging.info(f"trial_config:{trial_config}")
        logging.info(f"model_hparams:{model.hparams}")
        trainer = model_utils.get_trainer(trial_config, data_module)
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )
        optuna_logger.info(f"Trainer: {trainer}")
        optuna_logger.info(f"Trainer metrics {trainer.callback_metrics}")
        wandb.log(data={"validation loss": trainer.callback_metrics["val_loss"].item()})
        wandb.finish()
        # report result
        return trainer.callback_metrics["val_loss"].item()

    # setup optuna and run
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    )
    if study is None:
        study = optuna.create_study(
            direction="minimize", pruner=pruner, study_name=study_name
        )
    wandb_kwargs = {
        "project": "ats-optuna",
        "entity": "johnnychen7622",
        "reinit": True,
    }
    wandbc = WeightsAndBiasesCallback(metric_name="val_loss", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study
