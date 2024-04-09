import random

import numpy as np
try:
    import torch
    from torch import Tensor
except:
    print('Unable to import PyTorch.')


def to_tensor(x):
    if isinstance(x, np.ndarray):
        # If CUDA is available, convert array to tensor using torch.tensor, which honors the set default device.
        # Otherwise, use from_numpy which creates a reference to the data in memory instead of creating a copy.
        if torch.cuda.is_available():
            return torch.tensor(x)
        else:
            return torch.from_numpy(x)
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.asarray(x)
    else:
        return x


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except:
        pass


def interp1d_tensor(x0, y0, x):
    """
    1D interpolation for tensors.

    :param x0: Tensor. 1D tensor giving the coordinates.
    :param y0: Tensor. 1D tensor giving the values of the array to be interpolated.
    :param x: Tensor. 1D tensor giving the locations where new values are to be caluclated.
    :return: Tensor.
    """
    ibins = torch.bucketize(x, x0)
    y_floor = y0[torch.clamp(ibins - 1, 0, len(y0) - 1)]
    y_ceil = y0[torch.clamp(ibins, 0, len(y0) - 1)]
    x_floor = x0[torch.clamp(ibins - 1, 0, len(y0) - 1)]
    x_ceil = x0[torch.clamp(ibins, 0, len(y0) - 1)]
    w = ((x - x_floor) / (x_ceil - x_floor + 1e-8)).clamp(0, 1)
    vals = y_ceil * w + y_floor * (1 - w)
    return vals
