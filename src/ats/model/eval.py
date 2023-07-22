import logging
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE

from util import logging_utils

from math import ceil


def week_of_month(dt):
    """Returns the week of the month for the specified date."""
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom / 7.0))


if __name__ == "__main__":
    logging_utils.init_logging()

    data = pd.read_parquet("data/token/FUT/30min/ES", engine="fastparquet")
    data["Time"] = data.index
    data["ticker"] = "ES"
    data["volume"] = data["Volume"]
    data["close"] = data["Close"]
    data["Time"] = data["Time"].apply(lambda x: x.timestamp()).astype(np.float32)
    logging.info(f"data:{data.head()}")

    data["date"] = data.index
    # add time index
    data.insert(0, "time_idx", range(0, len(data)))
    # data["time_idx"] = data['date'].apply(lambda x:int(x.timestamp()))
    # data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_ticker"] = data.groupby(
        ["time_idx", "ticker"], observed=True
    ).volume.transform("mean")
    data["hour_of_day"] = (
        data["date"].apply(lambda x: x.hour).astype(str).astype("category")
    )
    data["day_of_week"] = data.index.dayofweek.astype(str).astype("category")
    data["day_of_month"] = data.index.day.astype(str).astype("category")
    data["week_of_month"] = (
        data["date"].apply(week_of_month).astype(str).astype("category")
    )
    data["week_of_year"] = data.index.isocalendar().week.astype(str).astype("category")
    logging.info(f"data:{data.head()}")
    logging.info(f"data:{data.describe()}")

    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    # data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    data.sample(10, random_state=521)

    max_prediction_length = 6
    max_encoder_length = 24
    val_idx = max(int(len(data) * 0.7), len(data) - 2048 * 16)
    tst_idx = max(int(len(data) * 0.8), len(data) - 2048)
    training_cutoff = val_idx
    train_data = data[:val_idx]
    logging.info(f"train_data:{train_data.head()}, training_cutoff={training_cutoff}")
    logging.info(f"training:{train_data.shape}")
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="close",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length
        // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        static_reals=[],
        allow_missing_timesteps=True,
        time_varying_known_categoricals=[
            "month",
            "hour_of_day",
            "day_of_week",
            "week_of_month",
        ],
        # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "close",
            "volume",
            "log_volume",
            "avg_volume_by_ticker",
        ],
        target_normalizer=GroupNormalizer(
            groups=["ticker"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    # configure network and trainer
    pl.seed_everything(42)
    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = (
        "lightning_logs/lightning_logs/version_1/checkpoints/epoch=19-step=1000.ckpt"
    )
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    logging.info(f"best_model_path:{best_model_path}")

    # calcualte mean absolute error on validation set
    # predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    # mae = MAE()(predictions.output, predictions.y)
    # logging.info(f"t:{mae}")

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    result = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    raw_predictions = result.output
    x = result.x
    logging.info(f"x:{x}")
    logging.info(f"raw_predictions:{raw_predictions}")
    for idx in range(10):  # plot 10 examples
        best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
    exit(0)

    # calcualte metric by which to display
    raw_predictions = best_tft.predict(
        val_dataloader, mode="raw", return_y=True, return_x=True
    )
    mean_losses = SMAPE(reduction="none")(
        raw_predictions.output, raw_predictions.y
    ).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    # logging.info(f"indices:{indices}")
    for idx in range(10):  # plot 10 examples
        best_tft.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=indices[idx],
            add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles),
        )
    exit(0)
    predictions = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
        predictions.x, predictions.output
    )
    logging.info(f"predictions_vs_actuals:{predictions_vs_actuals}")
    res = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    logging.info(f"result:{res}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    res = best_tft.predict(
        training.filter(lambda x: (x.time_idx_first_prediction == 15)),
        mode="quantiles",
    )
    logging.info(f"result:{res}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    raw_prediction = best_tft.predict(
        training.filter(lambda x: (x.time_idx_first_prediction == 15)),
        mode="raw",
        return_x=True,
    )
    best_tft.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0)

    # select last 24 months from data (max_encoder_length is 24)
    encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    # select last known data point and create decoder data from it by repeating it and incrementing the month
    # in a real world dataset, we should not just forward fill the covariates but specify them to account
    # for changes in special days and prices (which you absolutely should do but we are too lazy here)
    last_data = data[lambda x: x.time_idx == x.time_idx.max()]
    decoder_data = pd.concat(
        [
            last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i))
            for i in range(1, max_prediction_length + 1)
        ],
        ignore_index=True,
    )

    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)

    dependency = best_tft.predict_dependency(
        val_dataloader.dataset,
        "hour_of_day",
        np.linspace(0, 30, 30),
        show_progress_bar=True,
        mode="dataframe",
    )

    # plotting median and 25% and 75% percentile
    agg_dependency = dependency.groupby("hour_of_day").normalized_prediction.agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    ax = agg_dependency.plot(y="median")
    ax.fill_between(
        agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3
    )
