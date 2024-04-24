import os
import glob
import pickle
import argparse

import torch
import numpy as np
import pandas as pd

from autobl.steering.configs import *
from autobl.steering.acquisition import *
from autobl.steering.optimization import *
from autobl.steering.experiment import SimulatedScanningExperiment
from autobl.util import *


def test_xanes_experiment_manager(generate_gold=False, debug=False):
    torch.set_default_device('cpu')

    set_random_seed(123)

    data_path = 'data/xanes/Sample1_50C_XANES.csv'
    data_all_spectra = pd.read_csv(data_path, header=None)

    data = data_all_spectra.iloc[len(data_all_spectra) // 2].to_numpy()
    energies = data_all_spectra.iloc[0].to_numpy()
    energies = torch.tensor(energies)

    ref_spectra_0 = torch.tensor(data_all_spectra.iloc[1].to_numpy())
    ref_spectra_1 = torch.tensor(data_all_spectra.iloc[-1].to_numpy())
    ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
    ref_spectra_x = energies

    configs = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        override_kernel_lengthscale=7,
        noise_variance=1e-6,
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                     'differentiation_method': 'numerical',
                                     'reference_spectra_x': ref_spectra_x,
                                     'reference_spectra_y': ref_spectra_y,
                                     'phi_r': 1e3,
                                     'phi_g': 1e-2,
                                     'phi_g2': 1e-4,
                                     'addon_term_lower_bound': 3e-2,
                                     'debug': False},
        n_updates_create_acqf_weight_func=5,
        acqf_weight_func_floor_value=0.01,
        acqf_weight_func_post_edge_gain=3.0,
        optimizer_class=TorchOptimizer,
        optimizer_params={'torch_optimizer': torch.optim.Adam, 'torch_optimizer_options': {'maxiter': 20}}
    )

    experiment = SimulatedScanningExperiment(configs, run_analysis=False)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=10, n_target_measurements=80)

    # CI
    candidate_list = np.stack(experiment.candidate_list)
    gold_dir = os.path.join('gold_data', 'test_xanes_experiment_manager')
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
        assert np.allclose(candidate_list[:10], gold_data[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_xanes_experiment_manager(generate_gold=args.generate_gold, debug=True)
