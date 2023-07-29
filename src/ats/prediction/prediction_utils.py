import datetime
import logging

from pytorch_forecasting.utils import to_list
import torch

from ats.util import profile_util

day_of_week_map = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]

def loss_stats(pred_output):
    y_close_cum_sum_row = pred_output.y_close_cum_sum[pred_output.idx]
    y_close_cum_max = torch.max(y_close_cum_sum_row)
    y_close_cum_min = torch.min(y_close_cum_sum_row)
    y_hat = pred_output.y_hats[pred_output.idx]
    y_hat_cum = torch.cumsum(y_hat, dim=-1)
    y_hat_cum_max = torch.max(y_hat_cum)
    y_hat_cum_min = torch.min(y_hat_cum)    
    return y_close_cum_max, y_close_cum_min, y_hat_cum_max, y_hat_cum_min
    
def add_pred_context(env_mgr, matched_eval_data, idx, index, pred_input):
    config = env_mgr.config
    target_size = env_mgr.target_size
    context_length = env_mgr.context_length
    prediction_length = env_mgr.prediction_length
    train_data_row = matched_eval_data[
        matched_eval_data.time_idx == index.time_idx
    ].iloc[0]
    # logging.info(f"train_data_row:{train_data_row}")
    dm = train_data_row["time"]
    dm_str = datetime.datetime.strftime(dm, "%Y%m%d-%H%M%S")
    train_data_rows = matched_eval_data[
        (matched_eval_data.time_idx >= index.time_idx - config.model.context_length)
        & (matched_eval_data.time_idx < index.time_idx + config.model.prediction_length)
    ]
    decoder_time_idx = pred_input.x["decoder_time_idx"][idx][0].cpu().detach().numpy()
    x_time = matched_eval_data[
        (matched_eval_data.time_idx >= decoder_time_idx - context_length)
        & (matched_eval_data.time_idx < decoder_time_idx + prediction_length)
    ]["time"]
    prediction_date_time = (
        train_data_row["ticker"]
        + " "
        + dm_str
        + " "
        + day_of_week_map[train_data_row["day_of_week"]]
        + " "
        + str(train_data_row["close"])
    )
    pred_input.decoder_time_idx = decoder_time_idx
    pred_input.prediction_date_time = prediction_date_time
    pred_input.train_data_row = train_data_row
    pred_input.train_data_rows = train_data_rows
    pred_input.x_time = x_time

@profile_util.profile
def predict(model, new_prediction_data, wandb_logger, batch_size=1):
    # logging.info(f"new_prediction_data:{new_prediction_data}")
    # logging.info(f"index:{train_dataset.index.iloc[-5:]}")
    trainer_kwargs = {"logger": wandb_logger}
    # TODO: it is not clear why to_prediction fails complaining
    # about tensors on cpu even with to("cuda:0"). Maybe
    # something is going on with sampling ops which is placed on
    # cpu.
    #device = torch.device("cpu")
    #model.to(device)
    # logging.info(f"model:{model}, device:{model.device}")
    device = model.device
    new_raw_predictions = model.predict(
        new_prediction_data,
        mode="raw",
        return_x=True,
        batch_size=batch_size,
        trainer_kwargs=trainer_kwargs,
    )
    if isinstance(new_raw_predictions, (list)) and len(new_raw_predictions) < 1:
        logging.info(f"no prediction")
        return None, None
    prediction_kwargs = {}
    x = new_raw_predictions.x
    output = new_raw_predictions.output
    output = {
        key: [v.to(device) for v in val] if isinstance(val, list) else val.to(device)
        for key, val in output.items()
    }
    y_hats = model.to_prediction(output, **prediction_kwargs)
    quantiles_kwargs = {}
    y_quantiles = to_list(model.to_quantiles(output, **quantiles_kwargs))[0]
    del new_raw_predictions
    logging.info(f"y_quantiles:{y_quantiles}")
    logging.info(f"y_hats:{y_hats}")
    # logging.info(f"y:{new_raw_predictions.y}")
    return y_hats, y_quantiles, output, x
