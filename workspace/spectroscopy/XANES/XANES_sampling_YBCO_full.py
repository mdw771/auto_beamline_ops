import os
import glob
import pickle

import gpytorch.means
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
from autobl.steering.model import *
from autobl.steering.experiment import SimulatedScanningExperiment
from autobl.util import *

torch.set_default_device('cpu')

set_random_seed(124)

data_path = 'data/raw/YBCO/YBCO3data.csv'
data_all_spectra = pd.read_csv(data_path, header=0)
# # Only keep 8920 - 9080 eV
# data_all_spectra = data_all_spectra.iloc[14:232]

def linear_fit(basis_list, data):
    a = np.stack([to_numpy(ref_spectra_0), to_numpy(ref_spectra_1)]).T
    b = data.reshape(-1, 1)
    x = np.linalg.pinv(a) @ b
    y_fit = (a @ x).reshape(-1)
    return y_fit

data = data_all_spectra['YBCO_epararb.0001'].to_numpy()
ref_spectra_0 = torch.tensor(data_all_spectra['YBCO_epara.0001'].to_numpy())
ref_spectra_1 = torch.tensor(data_all_spectra['YBCO_eparc.0001'].to_numpy())
energies = data_all_spectra['energy'].to_numpy()
print('Energy range = {} eV'.format(energies[-1] - energies[0]))
energies = torch.tensor(energies)

ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

configs = XANESExperimentGuideConfig(
    dim_measurement_space=1,
    num_candidates=1,
    model_class=botorch.models.SingleTaskGP,
    # model_class=ProjectedSpaceSingleTaskGP,
    model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
    noise_variance=1e-6,
    override_kernel_lengthscale=7,
    lower_bounds=torch.tensor([energies[0]]),
    upper_bounds=torch.tensor([energies[-1]]),
    acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
    acquisition_function_params={'gradient_order': 2,
                                 'differentiation_method': 'numerical',
                                 'reference_spectra_x': ref_spectra_x,
                                 'reference_spectra_y': ref_spectra_y,
                                 'phi_r': None,
                                 'phi_g': None, #2e-2,
                                 'phi_g2': None, #3e-4
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
    # stopping_criterion_configs=StoppingCriterionConfig(
    #     method='max_uncertainty',
    #     params={'threshold': 0.01}
    # )
    project_func_sparseness_lower_bound=0.5,
    project_func_sparseness_plateau_bounds=(-5, 50),
    use_spline_interpolation_for_posterior_mean=True,
    debug=False
)

analyzer_configs = ExperimentAnalyzerConfig(
    name='YBCO3dataFull',
    output_dir='outputs',
    n_plot_interval=5,
    save=False,
    show=True
)

experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs,
                                         auto_narrow_down_scan_range=True, narrow_down_range_bounds_ev=(-100, 200))
experiment.build(energies, data)
experiment.run(n_initial_measurements=170, n_target_measurements=210)

x = np.linspace(energies[0], energies[-1], 1000)
y = experiment.get_estimated_spectrum(x)
plt.plot(x, y)
plt.show()
