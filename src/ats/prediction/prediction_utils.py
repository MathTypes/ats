import logging

from pytorch_forecasting.utils import create_mask, detach, to_list
import torch

def predict(model, new_prediction_data, wandb_logger):
    # logging.info(f"new_prediction_data:{new_prediction_data}")
    #logging.info(f"index:{train_dataset.index.iloc[-5:]}")
    trainer_kwargs = {"logger": wandb_logger}
    # logging.info(f"trainer_kwargs:{trainer_kwargs}")
    # TODO: it is not clear why to_prediction fails complaining
    # about tensors on cpu even with to("cuda:0"). Maybe
    # something is going on with sampling ops which is placed on
    # cpu.
    device = torch.device('cpu')
    model.to(device)
    #logging.info(f"model:{model}, device:{model.device}")
    device = model.device
    new_raw_predictions = model.predict(
        new_prediction_data,
        mode="raw",
        return_x=True,
        batch_size=1,
        trainer_kwargs=trainer_kwargs,
    )
    #logging.info(f"new_raw_predictions:{new_raw_predictions}")
    if isinstance(new_raw_predictions, (list)) and len(new_raw_predictions) < 1:
        logging.info(f"no prediction")
        return None, None
    prediction_kwargs = {}
    output = new_raw_predictions.output
    output = {
        key: [v.to(device) for v in val]
        if isinstance(val, list)
        else val.to(device)
        for key, val in output.items()
    }
    y_hats = to_list(
        model.to_prediction(output, **prediction_kwargs)
    )
    #logging.info(f"y_hats:{y_hats}")
    prediction = y_hats
    #logging.info(f"prediction:{prediction}")
    quantiles_kwargs = {}
    y_quantiles = to_list(
        model.to_quantiles(output, **quantiles_kwargs)
    )[0]
    del new_raw_predictions
    #logging.info(f"y_quantiles:{y_quantiles}")
    # logging.info(f"y_hats:{y_hats}")
    # logging.info(f"y:{new_raw_predictions.y}")
    return prediction, y_quantiles, output
