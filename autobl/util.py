import random

import numpy as np
try:
    import torch
except:
    print('Unable to import PyTorch.')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except:
        pass
