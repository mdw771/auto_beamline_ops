import os
import glob
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib
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
from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools


matplotlib.rc('font', family='Times New Roman')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['pdf.fonttype'] = 42

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

normalizer = xanestools.XANESNormalizer()
normalizer.fit(energies, data, fit_ranges=((11400, 11500), (11650, 11850)))

# Only keep 11400 - 11700 eV
mask = (energies >= 11400) & (energies <= 11700)
data = data[mask]
ref_spectra_0 = ref_spectra_0[mask]
ref_spectra_1 = ref_spectra_1[mask]
energies = energies[mask]

energies = torch.tensor(energies)

ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

configs = XANESExperimentGuideConfig(
    dim_measurement_space=1,
    num_candidates=1,
    model_class=botorch.models.SingleTaskGP,
    model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
    noise_variance=1e-6,
    # override_kernel_lengthscale=7,
    lower_bounds=torch.tensor([energies[0]]),
    upper_bounds=torch.tensor([energies[-1]]),
    acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
    acquisition_function_params={'gradient_order': 2,
                                 'differentiation_method': 'numerical',
                                 'reference_spectra_x': ref_spectra_x,
                                 'reference_spectra_y': ref_spectra_y,
                                 'phi_r': 1e2,
                                 'phi_g': 2e-3, #2e-2,
                                 'phi_g2': 2e-3, #3e-4
                                 'beta': 0.999,
                                 'gamma': 0.95,
                                 'addon_term_lower_bound': 3e-2,
                                 'estimate_posterior_mean_by_interpolation': True,
                                 'subtract_background_gradient': True,
                                 'debug': False
                                 },

    optimizer_class=DiscreteOptimizer,
    optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                      'optim_func_params': {
                          'choices': torch.linspace(0, 1, 1000)[:, None]
                      }
                     },

    n_updates_create_acqf_weight_func=1,
    acqf_weight_func_floor_value=0.01,
    acqf_weight_func_post_edge_gain=3.0,
    acqf_weight_func_post_edge_offset=2.0,
    acqf_weight_func_post_edge_width=1.0,
    stopping_criterion_configs=StoppingCriterionConfig(
        method='max_uncertainty',
        params={'threshold': 0.01}
    ),
    use_spline_interpolation_for_posterior_mean=True
)

experiment = SimulatedScanningExperiment(configs, run_analysis=True)
experiment.build(energies, data)
experiment.run(n_initial_measurements=20, n_target_measurements=21, initial_measurement_method='quasirandom')

x_norm = torch.linspace(0.01, 0.99, 100).reshape(-1, 1, 1)
mu, sigma = experiment.guide.acquisition_function._mean_and_sigma(x_norm)
y_acqf = experiment.guide.acquisition_function(x_norm)
y_uncertainty = sigma.squeeze()
y_grad = experiment.guide.acquisition_function.acqf_g(x_norm, mu_x=mu, sigma_x=sigma)
y_residue = experiment.guide.acquisition_function.acqf_r(x_norm, mu_x=mu, sigma_x=sigma)

# experiment = SimulatedScanningExperiment(configs, run_analysis=True)
# experiment.build(energies, data)
# experiment.run(n_initial_measurements=20, n_target_measurements=25, initial_measurement_method='quasirandom')
y_weight = experiment.guide.acqf_weight_func(x_norm.reshape(-1, 1))

mask = y_acqf.squeeze().detach().cpu().numpy() > 1e-8
plt.semilogy(x_norm.squeeze().detach().cpu().numpy()[mask], y_acqf.squeeze().detach().cpu().numpy()[mask] * 1e4, linewidth=2, label='Acqusition')
plt.semilogy(x_norm.squeeze().detach().cpu().numpy()[mask], y_uncertainty.squeeze().detach().cpu().numpy()[mask], linewidth=1, label='Uncertainty')
plt.semilogy(x_norm.squeeze().detach().cpu().numpy()[mask], y_grad.squeeze().detach().cpu().numpy()[mask], linewidth=1, label='Gradient')
plt.semilogy(x_norm.squeeze().detach().cpu().numpy()[mask], y_residue.squeeze().detach().cpu().numpy()[mask], linewidth=1, label='Residue')
plt.semilogy(x_norm.squeeze().detach().cpu().numpy()[mask], y_weight.squeeze().detach().cpu().numpy()[mask], linewidth=1, label='Reweighting')
plt.legend()
plt.xticks([])
plt.yticks([])
plt.savefig("factory/acquisition_function.pdf", bbox_inches='tight')


