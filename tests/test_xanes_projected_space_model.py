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
from autobl.steering.model import ProjectedSpaceSingleTaskGP
from autobl.util import *


def test_xanes_projected_space_model(generate_gold=False, debug=False):
    torch.set_default_device('cpu')

    set_random_seed(124)

    data_path = 'data/xanes/YBCO3data.csv'
    data_all_spectra = pd.read_csv(data_path, header=0)

    data = data_all_spectra['YBCO_epararb.0001'].to_numpy()
    ref_spectra_0 = torch.tensor(data_all_spectra['YBCO_epara.0001'].to_numpy())
    ref_spectra_1 = torch.tensor(data_all_spectra['YBCO_eparc.0001'].to_numpy())
    energies = data_all_spectra['energy'].to_numpy()
    energies = torch.tensor(energies)

    ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
    ref_spectra_x = energies

    configs = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=ProjectedSpaceSingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        noise_variance=1e-6,
        override_kernel_lengthscale=14,
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                     'differentiation_method': 'numerical',
                                     'reference_spectra_x': ref_spectra_x,
                                     'reference_spectra_y': ref_spectra_y,
                                     'phi_r': None,
                                     'phi_g': None,  # 2e-2,
                                     'phi_g2': None,  # 3e-4
                                     'beta': 0.999,
                                     'gamma': 0.99,
                                     'addon_term_lower_bound': 3e-2,
                                     'debug': False
                                     },

        optimizer_class=DiscreteOptimizer,
        optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                          'optim_func_params': {
                              'choices': torch.linspace(0, 1, 5000)[:, None]
                          }
                          },

        n_updates_create_acqf_weight_func=5,
        acqf_weight_func_floor_value=0.01,
        acqf_weight_func_post_edge_gain=3.0,
    )

    experiment = SimulatedScanningExperiment(configs, run_analysis=True)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=140, n_target_measurements=180)

    # CI
    candidate_list = np.stack(experiment.candidate_list)
    gold_dir = os.path.join('gold_data', 'test_xanes_projected_space_model')
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

    test_xanes_projected_space_model(generate_gold=args.generate_gold, debug=True)
