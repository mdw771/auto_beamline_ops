import os
import argparse

import gpytorch.kernels
import numpy as np
import botorch
import torch

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.steering.acquisition import *
from autobl.util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    set_random_seed(123)
    candidate_list = []

    # Define a simulated measurement function
    def measurement_func(x: torch.Tensor):
        y = -(x - 0.6) ** 2
        y = y + 0.001 * torch.randn_like(y)
        return y
    measurement = SimulatedMeasurement(f=measurement_func)

    # Generate test data
    train_x = torch.rand(10, 1).double()
    train_y = measurement.measure(train_x)

    config = GPExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(1.5)},
        acquisition_function_class=GradientAwarePosteriorStandardDeviation,
        acquisition_function_params={'phi': 0.01},
        lower_bounds=torch.zeros(train_x.shape[1]),
        upper_bounds=torch.ones(train_x.shape[1])
    )
    guide = autobl.steering.guide.GPExperimentGuide(config)
    guide.build(train_x, train_y)
    candidates = guide.suggest()
    print(candidates)
    candidate_list.append(candidates.squeeze().detach().cpu().numpy())
    guide.plot_posterior(torch.linspace(0, 1, 100))

    for i in range(5):
        guide.update(candidates, measurement.measure(candidates))
        candidates = guide.suggest()
        print(candidates)
        candidate_list.append(candidates.squeeze().detach().cpu().numpy())
        guide.plot_posterior(torch.linspace(0, 1, 100))
