import random

import numpy as np
try:
    import torch
except:
    print('Unable to import PyTorch.')


def to_tensor(x):
    if isinstance(x, np.ndarray):
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
