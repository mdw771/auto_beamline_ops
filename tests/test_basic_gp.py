import gpytorch.kernels
import numpy as np
import botorch
import torch

import autobl.steering
from autobl.steering.configs import *


def test_basic_gp(generate_gold=False):
    # Generate test data
    train_x = torch.rand(10, 1)
    train_y = -(train_x - 0.6) ** 2
    train_y = train_y + 0.001 * torch.randn_like(train_y)
    # train_y = botorch.utils.standardize(train_y)

    config = GPExperimentGuideConfig(
        measurement_space_dims=[100],
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(1.5)},
        acquisition_function_class=botorch.acquisition.UpperConfidenceBound,
        acquisition_function_params={'beta': 0.1},
        lower_bounds=torch.zeros(train_x.shape[1]),
        upper_bounds=torch.ones(train_x.shape[1])
    )
    guide = autobl.steering.guide.GPExperimentGuide(config)
    guide.build((train_x, train_y))
    candidates = guide.suggest()
    print(candidates)
    guide.plot_posterior(torch.linspace(0, 1, 100))


if __name__ == '__main__':
    test_basic_gp()
