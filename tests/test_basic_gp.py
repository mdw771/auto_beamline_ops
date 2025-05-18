import os
import argparse

import gpytorch.kernels
import numpy as np
import botorch
import torch

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.util import *


def test_basic_gp(generate_gold=False, debug=False):
    torch.set_default_device('cpu')
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
        acquisition_function_class=botorch.acquisition.PosteriorStandardDeviation,
        # acquisition_function_params={'beta': 0.1},
        lower_bounds=torch.zeros(train_x.shape[1]),
        upper_bounds=torch.ones(train_x.shape[1])
    )
    guide = autobl.steering.guide.GPExperimentGuide(config)
    guide.build(train_x, train_y)
    candidates = guide.suggest()
    print(candidates)
    candidate_list.append(candidates.squeeze().detach().cpu().numpy())
    if debug:
        guide.plot_posterior(torch.linspace(0, 1, 100))

    for i in range(5):
        guide.update(candidates, measurement.measure(candidates))
        candidates = guide.suggest()
        print(candidates)
        candidate_list.append(candidates.squeeze().detach().cpu().numpy())
        if debug:
            guide.plot_posterior(torch.linspace(0, 1, 100))

    # CI
    candidate_list = np.stack(candidate_list)
    gold_dir = os.path.join('gold_data', 'test_basic_gp')
    if generate_gold:
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
        np.save(os.path.join(gold_dir, 'candidates.npy'), candidate_list)

    if not debug:
        gold_data = np.load(os.path.join(gold_dir, 'candidates.npy'), allow_pickle=True)
        print('=== Current ===')
        print(candidate_list)
        print('=== Reference ===')
        print(gold_data)
        
        # There is a problem with GitHub hosted CI; we need to move to self-hosted before enabling it.
        if False:
            assert np.allclose(candidate_list, gold_data, rtol=0.05)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_basic_gp(generate_gold=args.generate_gold, debug=~args.generate_gold)
