import torch

from ats.util import tensor_utils

def get_embedding(self, data_module, model: MetricModel, ticker, timestamp) -> list[float]:
    """Perform single inference of image with proper model and return embeddings"""
    model = model.model
    full_data = data_module.full_data
    full_data = full_data[full_data.ticker.isin(["ES"])]
    raw_data = full_data[
        (full_data.timestamp == row["timestamp"])
        & (full_data.ticker == row["ticker"])
    ]
    decoder_time_idx=raw_data["time_idx"]
    logging.error(f"decoder_time_idx:{decoder_time_idx}")
    eval_dataset = data_module.validation
    filtered_dataset = eval_dataset.filter(
        lambda x: x.time_idx_first_prediction==decoder_time_idx+1
    )
    with torch.no_grad():
        val_x, val_y = next(iter(filtered_dataset.to_dataloader()))
        logging.error(f"val_x:{val_x}")
        logging.error(f"val_y:{val_y}")
        if torch.cuda.is_available():
            output = model(tensor_utils.to_cuda(val_x))
        else:
            output = model(val_x)
        logging.error(f"output:{output}")
        embedding = output["embedding"]
        embedding = torch.reshape(embedding, (embedding.size(0), -1))
        embedding = embedding[0]
        logging.error(f"embedding:{embedding.shape}")
    return embedding
