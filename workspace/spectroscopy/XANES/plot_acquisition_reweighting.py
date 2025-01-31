import os
import glob
import pickle

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
from autobl.util import *
from autobl.steering.guide import *
from autobl.steering.io_util import LTORawDataset
import autobl.tools.spectroscopy.xanes as xanestools


matplotlib.rc("font", family="Times New Roman")
matplotlib.rcParams["font.size"] = 18
matplotlib.rcParams["pdf.fonttype"] = 42

torch.set_default_device('cpu')

set_random_seed(123)

dataset = LTORawDataset('data/raw/LiTiO_XANES/rawdata', filename_pattern="LTOsample1.[0-9]*")
data_all_spectra = dataset.data
energies = dataset.energies_ev
data = data_all_spectra[len(data_all_spectra) // 2]

mask = (energies >= 4936) & (energies <= 5006)
data_all_spectra = data_all_spectra[:, mask]
data = data_all_spectra[len(data_all_spectra) // 2]
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
        params={'threshold': 0.0001}
    ),
    use_spline_interpolation_for_posterior_mean=True
)

experiment = SimulatedScanningExperiment(configs, run_analysis=False)
experiment.build(energies, data)
experiment.run(n_initial_measurements=10, n_target_measurements=15, initial_measurement_method='quasirandom')

guide = experiment.guide
y_acqf = guide.acqf_weight_func(torch.linspace(0, 1, len(energies)).reshape(-1, 1))

x_posterior = energies.reshape(-1, 1)
y_posterior, _ = guide.get_posterior_mean_and_std(x_posterior, transform=True, untransform=True)

fig, ax = plt.subplots(1, 1)
ax.plot(energies.squeeze(), y_acqf.squeeze(), label="Reweighting\n function")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Reweighting function w(x)")
ax2 = ax.twinx()
ax2.plot(x_posterior.squeeze(), y_posterior.squeeze(), label="Posterior\n mean", color="gray", linestyle="--")
ax2.set_ylabel("Absorption")
fig.legend(bbox_to_anchor=(0.48, 0.87), frameon=False, ncol=1, fontsize=16)


ax2.plot(energies.squeeze(), data.squeeze(), label="Ground truth", color="red", linestyle="--")

plt.show()
# plt.savefig("factory/acqf_reweighting.pdf", bbox_inches='tight')
