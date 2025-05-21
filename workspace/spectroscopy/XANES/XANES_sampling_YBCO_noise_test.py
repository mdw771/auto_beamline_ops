import os
import glob
import pickle
import sys
import logging

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import tqdm

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.steering.acquisition import *
from autobl.steering.optimization import *
from autobl.steering.experiment import SimulatedScanningExperiment
from autobl.steering.guide import *
from autobl.steering.io_util import *
from autobl.tools.spectroscopy.xanes import XANESNormalizer
from autobl.util import *
from autobl.steering.util import estimate_noise_variance

torch.set_default_device('cpu')

set_random_seed(124)

def linear_fit(basis_list, data):
    a = np.stack([to_numpy(ref_spectra_0), to_numpy(ref_spectra_1)]).T
    b = data.reshape(-1, 1)
    x = np.linalg.pinv(a) @ b
    y_fit = (a @ x).reshape(-1)
    return y_fit

dataset = YBCORawDataset('data/raw/YBCO/YBCO_epararb.0001')
data = dataset[0]
energies = dataset.energies_ev
ref_spectra_0 = YBCORawDataset('data/raw/YBCO/YBCO_epara.0001')[0]
ref_spectra_1 = YBCORawDataset('data/raw/YBCO/YBCO_eparc.0001')[0]

# Fit detilter and normalizer
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(to_numpy(energies), data, label='data')
plt.show()
xanes_normalizer = XANESNormalizer(fit_ranges=((8788, 8914), (9401, 10000)), edge_loc=8990)
xanes_normalizer.fit(energies, data)

# Only keep 8920 - 9080 eV
data = data[14:232]
ref_spectra_0 = torch.tensor(ref_spectra_0[14:232])
ref_spectra_1 = torch.tensor(ref_spectra_1[14:232])
energies = energies[14:232]
energies = torch.tensor(energies)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(to_numpy(energies), data, label='data')
plt.show()

# Add noise
added_noise_variance = 1e-4

ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

n_passes = 1
for noise_var in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    configs = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        noise_variance=noise_var,
        adaptive_noise_variance=True,
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        reference_spectra_for_lengthscale_fitting=(ref_spectra_x, ref_spectra_y[1]),
        acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                    'differentiation_method': 'numerical',
                                    'reference_spectra_x': ref_spectra_x,
                                    'reference_spectra_y': ref_spectra_y,
                                    'phi_r': None,
                                    'phi_g': None, #2e-2,
                                    'phi_g2': None, #3e-4
                                    'beta': 0.999,
                                    'gamma': 0.95,
                                    'addon_term_lower_bound': 3e-2,
                                    'estimate_posterior_mean_by_interpolation': False,
                                    'debug': False
                                    },

        optimizer_class=DiscreteOptimizer,
        optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                        'optim_func_params': {
                            'choices': torch.linspace(0, 1, 1000)[:, None]
                        }
                        },

        n_updates_create_acqf_weight_func=5,
        acqf_weight_func_floor_value=0.01,
        acqf_weight_func_post_edge_gain=3.0,
        acqf_weight_func_post_edge_offset=2.0,
        acqf_weight_func_post_edge_width=1.0,
        stopping_criterion_configs=StoppingCriterionConfig(
            method='max_uncertainty',
            params={'threshold': 0},
            n_max_measurements=100
        ),
        use_spline_interpolation_for_posterior_mean=False
    )

    analyzer_configs = ExperimentAnalyzerConfig(
        name='YBCO3data',
        output_dir='outputs/YBCO_raw_randInit',
        n_plot_interval=5
    )

    set_random_seed(124)
    pass_str = 'noiseVar_{:.1e}'.format(noise_var)
    analyzer_configs.output_dir = 'outputs/YBCO_raw_randInit_addedNoiseVar_{:.1e}_{}'.format(added_noise_variance, pass_str)
    
    experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs, instrument_noise_var=added_noise_variance)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=20, n_target_measurements=70, initial_measurement_method='random')
    xanes_normalizer.save_state('outputs/YBCO_raw_randInit_addedNoiseVar_{:.1e}_{}/normalizer_state.npy'.format(added_noise_variance, pass_str))
