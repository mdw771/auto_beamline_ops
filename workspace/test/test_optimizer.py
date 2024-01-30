import numpy as np
import torch

import autobl.optimizer


def test_pytorch_optimizer():
    x = np.array([0.0, 0.0])
    opt = autobl.optimizer.PyTorchOptimizer(x, torch.optim.SGD, {'lr': 0.1})
    opt.step(np.array([-1.0, -1.0]))
    assert np.allclose(opt.x, np.array([0.1, 0.1]))


if __name__ == '__main__':
    test_pytorch_optimizer()