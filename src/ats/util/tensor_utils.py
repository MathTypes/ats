
import numpy as np
import torch

def to_cuda(val_x):
    if isinstance(val_x, (list)):
        val_x = [to_cuda(val) for val in val_x]
    elif isinstance(val_x, (dict)):
        val_x = {key:to_cuda(val) for key, val in val_x.items()}
    elif isinstance(val_x, (np.int64, np.ndarray)):
        valx_x = torch.tensor(val_x).cuda()
    else:
        val_x = val_x.cuda()
    return val_x

def np_to_native(val_x):
    if isinstance(val_x, (list)):
        val_x = [np_to_native(val) for val in val_x]
    elif isinstance(val_x, (dict)):
        val_x = {key:np_to_native(val) for key, val in val_x.items()}
    elif isinstance(val_x, (np.int64, np.float32)):
        val_x = val_x.item()
    else:
        val_x = val_x
    return val_x
