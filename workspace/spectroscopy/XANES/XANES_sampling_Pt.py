import os
import glob
import pickle
import sys

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
from autobl.steering.experiment import SimulatedUniformSamplingExperiment, SimulatedScanningExperiment
from autobl.steering.guide import *
from autobl.steering.io_util import *
from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from autobl.steering.util import estimate_noise_variance


torch.set_default_device('cpu')

set_random_seed(124)

data_raw = read_nor('data/raw/Pt-XANES/Pt_xmu/10Feb_PtL3_025_042C.xmu')
_, unique_inds = np.unique(data_raw['e'], return_index=True)
unique_inds = np.sort(unique_inds)
data_raw = data_raw.iloc[unique_inds]
ref1 = read_nor('data/raw/Pt-XANES/Pt_xmu/10Feb_PtL3_024_026C.xmu').iloc[unique_inds]
ref2 = read_nor('data/raw/Pt-XANES/Pt_xmu/10Feb_PtL3_045_497C.xmu').iloc[unique_inds]

def linear_fit(basis_list, data):
    a = np.stack([to_numpy(ref_spectra_0), to_numpy(ref_spectra_1)]).T
    b = data.reshape(-1, 1)
    x = np.linalg.pinv(a) @ b
    y_fit = (a @ x).reshape(-1)
    return y_fit

data = data_raw['xmu'].to_numpy()
ref_spectra_0 = torch.tensor(ref1['xmu'].to_numpy())
ref_spectra_1 = torch.tensor(ref2['xmu'].to_numpy())
energies = data_raw['e'].to_numpy()

if False:
    plt.plot(to_numpy(energies), to_numpy(data))
    plt.show()

normalizer = xanestools.XANESNormalizer()
normalizer.fit(energies, data, fit_ranges=((11400, 11500), (11650, 11850)))

# Only keep 11400 - 11700 eV
mask = (energies >= 11400) & (energies <= 11700)
data = data[mask]
ref_spectra_0 = ref_spectra_0[mask]
ref_spectra_1 = ref_spectra_1[mask]
energies = energies[mask]

# Estimate noise standard deviation
noise_std = estimate_noise_variance(to_numpy(energies[energies < 11500]), to_numpy(ref_spectra_0[energies < 11500]))
print(f'Noise standard deviation: {noise_std}')

energies = torch.tensor(energies)
if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(to_numpy(energies), data, label='data')
    plt.legend()
    plt.show()

ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

n_passes = 1
for i_pass in range(n_passes):
    configs = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        noise_variance=noise_std ** 2,
        reference_spectra_for_lengthscale_fitting=(ref_spectra_x, ref_spectra_y[0]),
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                    'differentiation_method': 'numerical',
                                    'reference_spectra_x': ref_spectra_x,
                                    'reference_spectra_y': ref_spectra_y,
                                    'phi_r': 1e2,
                                    'phi_g': 2e-3, #2e-2,
                                    'phi_g2': 1, #3e-4
                                    'beta': 0.999,
                                    'gamma': 0.95,
                                    'addon_term_lower_bound': 3e-2,
                                    'estimate_posterior_mean_by_interpolation': False,
                                    'subtract_background_gradient': True,
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
            n_max_measurements=60,
            params={'threshold': 0.005}
        ),
        use_spline_interpolation_for_posterior_mean=True
    )

    analyzer_configs = ExperimentAnalyzerConfig(
        name='Pt',
        output_dir='outputs/Pt_raw_uniInit_bgGrad',
        n_plot_interval=5
    )

    if n_passes > 1:
        set_random_seed(124 + i_pass)
    pass_str = f'_pass{i_pass}' if n_passes > 1 else ''
    
    analyzer_configs.output_dir = f'outputs/Pt_raw_uniInit_bgGrad{pass_str}'
    normalizer.save_state(f'outputs/Pt_raw_uniInit_bgGrad{pass_str}/normalizer_state.npy')
    experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')


    if True:
        # No acquisition reweighting
        set_random_seed(124 + i_pass)
        configs.n_updates_create_acqf_weight_func = None
        analyzer_configs.output_dir = f'outputs/Pt_raw_uniInit_noReweighting_bgGrad{pass_str}'
        experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(energies, data)
        experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')

        
        # Posterior standard deviation-only acquisition
        set_random_seed(124 + i_pass)
        configs.acquisition_function_class = PosteriorStandardDeviation
        configs.acquisition_function_params = {}
        configs.stopping_criterion_configs = None
        analyzer_configs.output_dir = f'outputs/Pt_raw_uniInit_posteriorStddev_bgGrad{pass_str}'
        experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(energies, data)
        experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')
        
        # UCB
        set_random_seed(124 + i_pass)
        configs.acquisition_function_class = UpperConfidenceBound
        configs.acquisition_function_params = {"beta": 10 ** 2}
        configs.stopping_criterion_configs = None
        analyzer_configs.output_dir = f'outputs/Pt_raw_uniInit_UCB_kappa_10{pass_str}'
        experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(energies, data)
        experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')

        # Uniform sampling
        set_random_seed(124 + i_pass)
        configs.n_updates_create_acqf_weight_func = None
        configs.stopping_criterion_configs = None
        analyzer_configs.output_dir = f'outputs/Pt_raw_uniInit_uniformSampling_bgGrad{pass_str}'
        experiment = SimulatedUniformSamplingExperiment(configs, guide_class=UniformSamplingExperimentGuide,
                                                run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(energies, data)
        experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')
