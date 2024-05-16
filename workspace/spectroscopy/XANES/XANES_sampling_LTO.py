import os
import glob
import pickle

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

torch.set_default_device('cpu')

set_random_seed(123)

data_path = 'data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample1_50C_XANES.csv'
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
                                 'phi_r': None, #1e3,
                                 'phi_g': None, #1e-2,
                                 'phi_g2': None, #1e-4,
                                 'beta': 0.999,
                                 'gamma': 0.95,
                                 'addon_term_lower_bound': 3e-2,
                                 'debug': False},
    n_updates_create_acqf_weight_func=5,
    acqf_weight_func_floor_value=0.01,
    acqf_weight_func_post_edge_gain=3.0,
    
    # optimizer_class=ContinuousOptimizer,
    # optimizer_params={'num_restarts': 2,}
    #                   #'options': {'maxiter': 2}}
    
    optimizer_class=DiscreteOptimizer,
    optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                      'optim_func_params': {
                          'choices': torch.linspace(0, 1, 1000)[:, None]
                      }
                     },
    
    # optimizer_class=TorchOptimizer,
    # optimizer_params={'torch_optimizer': torch.optim.Adam, 'torch_optimizer_options': {'maxiter': 20}},
    stopping_criterion_configs=StoppingCriterionConfig(
        method='max_uncertainty',
        params={'threshold': 0.05}
    )
)

analyzer_configs = ExperimentAnalyzerConfig(
    name='Sample1_50C_XANES',
    output_dir='outputs',
    n_plot_interval=5
)

experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
experiment.build(energies, data)
experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='random')
