import os
import glob
import pickle
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
from autobl.util import *
from autobl.steering.guide import *
from autobl.steering.io_util import LTORawDataset
import autobl.tools.spectroscopy.xanes as xanestools
from autobl.steering.util import estimate_noise_variance

torch.set_default_device('cpu')

set_random_seed(123)

dataset = LTORawDataset('data/raw/LiTiO_XANES/rawdata', filename_pattern="LTOsample3.[0-9]*")
data_all_spectra = dataset.data
energies = dataset.energies_ev
data = data_all_spectra[91]

plt.figure()
plt.plot(energies, data)
plt.show()

normalizer = xanestools.XANESNormalizer()
normalizer.fit(energies, data, fit_ranges=((4900, 4950), (5100, 5200)))

noise_variance = estimate_noise_variance(to_numpy(energies[energies < 4960]), to_numpy(data_all_spectra[0][energies < 4960]))
print("noise_variance: {}".format(noise_variance))

mask = (energies >= 4936) & (energies <= 5006)
data_all_spectra = data_all_spectra[:, mask]
data = data_all_spectra[91]
energies = energies[mask]
energies = torch.tensor(energies)
ref_spectra_0 = torch.tensor(data_all_spectra[0])
ref_spectra_1 = torch.tensor(data_all_spectra[-1])
ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

configs = XANESExperimentGuideConfig(
    dim_measurement_space=1,
    num_candidates=1,
    model_class=botorch.models.SingleTaskGP,
    model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
    reference_spectra_for_lengthscale_fitting=(
        to_tensor(ref_spectra_x), 
        to_tensor(ref_spectra_y[0])
    ),
    noise_variance=noise_variance,
    lower_bounds=torch.tensor([energies[0]]),
    upper_bounds=torch.tensor([energies[-1]]),
    acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
    acquisition_function_params={
        'gradient_order': 2,
        'differentiation_method': 'numerical',
        'reference_spectra_x': ref_spectra_x,
        'reference_spectra_y': ref_spectra_y,
        'phi_r': 1e4,
        'phi_g': 1e-1,
        'phi_g2': 1e-3,
        'beta': 0.999,
        'gamma': 0.95,
        'addon_term_lower_bound': 3e-2,
        'debug': False
    },
    n_updates_create_acqf_weight_func=5,
    acqf_weight_func_floor_value=0.01,
    acqf_weight_func_post_edge_gain=3.0,
    acqf_weight_func_post_edge_offset=2.0,
    acqf_weight_func_post_edge_width=1.0,
    
    optimizer_class=DiscreteOptimizer,
    optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                      'optim_func_params': {
                          'choices': torch.linspace(0, 1, 1000)[:, None]
                      }
                     },
    
    stopping_criterion_configs=StoppingCriterionConfig(
        method='max_uncertainty',
        params={'threshold': 0.0}
    ),
    use_spline_interpolation_for_posterior_mean=True
)

analyzer_configs = ExperimentAnalyzerConfig(
    name='Sample1_50C_XANES',
    output_dir='outputs/LTO_raw_uniInit',
    n_plot_interval=5
)

normalizer.save_state('outputs/LTO_raw_uniInit/normalizer_state.npy')

experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
experiment.build(energies, data)
experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform')

if False:
    set_random_seed(124)
    configs.n_updates_create_acqf_weight_func = None
    analyzer_configs.output_dir = 'outputs/LTO_raw_randInit_noReweighting'
    experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='random')

    set_random_seed(124)
    configs.acquisition_function_class = PosteriorStandardDeviation
    configs.acquisition_function_params = {}
    configs.stopping_criterion_configs = None
    analyzer_configs.output_dir = 'outputs/LTO_raw_randInit_posteriorStddev'
    experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='random')

    # Uniform sampling
    set_random_seed(124)
    configs.n_updates_create_acqf_weight_func = None
    configs.stopping_criterion_configs = None
    analyzer_configs.output_dir = 'outputs/LTO_raw_randInit_uniformSampling'
    experiment = SimulatedScanningExperiment(configs, guide_class=UniformSamplingExperimentGuide,
                                            run_analysis=True, analyzer_configs=analyzer_configs)
    experiment.build(energies, data)
    experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='random')
