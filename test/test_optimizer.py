import numpy as np
import torch

import program.optimizer


def test_pytorch_optimizer():
    x = np.array([0.0, 0.0])
    opt = program.optimizer.PyTorchOptimizer(x, torch.optim.SGD, {'lr': 0.1})
    opt.step(np.array([-1.0, -1.0]))
    assert np.allclose(opt.x, np.array([0.1, 0.1]))


if __name__ == '__main__':
    test_pytorch_optimizer()