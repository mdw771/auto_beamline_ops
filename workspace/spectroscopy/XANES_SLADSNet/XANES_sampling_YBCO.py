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
from autobl.steering.experiment import SimulatedScanningExperiment
from autobl.steering.guide import *
from autobl.steering.model import *
from autobl.steering.io_util import *
from autobl.tools.spectroscopy.xanes import XANESNormalizer
from autobl.util import *

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
# y_fit = linear_fit([to_numpy(ref_spectra_0), to_numpy(ref_spectra_1)], data)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(to_numpy(energies), data, label='data')
# ax.plot(to_numpy(energies), to_numpy(ref_spectra_0), label='ref1')
# ax.plot(to_numpy(energies), to_numpy(ref_spectra_1), label='ref2')
# ax.plot(to_numpy(energies), y_fit, label='fit', linestyle='--')
# plt.legend()
plt.show()

ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

model_params = {
    "dim_recon_spec": 1000,
    "dim_spec_encoded": 256,
    "dim_feat_encoded": 256,
    "dim_hidden_feat": 128,
    "dim_hidden_final": 128,
    "add_pooling": True,
    "sigmoid_on_output": False
}

configs = SLDAS1DExperimentGuideConfig(
    dim_measurement_space=1,
    num_candidates=1,
    model_class=ConvMLPModel,
    model_params=model_params,
    model_path="trained_models/model_normalizedERD_convMLP_dimReconSpec1000_dimSpecEncoded256_dimFeatEncoded256_dimHiddenFeat128_dimHiddenFinal128_pooled_featPos3NNs_gaussRandEnc_lr1e-3/final_model.pth",
    lower_bounds=torch.tensor([energies[0]]),
    upper_bounds=torch.tensor([energies[-1]]),
    n_eval_pixels=1000,
    debug=False
)

analyzer_configs = ExperimentAnalyzerConfig(
    name='YBCO3data',
    output_dir='outputs/YBCO_raw_randInit',
    n_plot_interval=5
)

experiment = SimulatedScanningExperiment(configs, guide_class=SLDAS1DExperimentGuide, run_analysis=True, analyzer_configs=analyzer_configs)
experiment.build(energies, data)
experiment.run(n_initial_measurements=20, n_target_measurements=70, initial_measurement_method='random')
xanes_normalizer.save_state('outputs/YBCO_raw_randInit/normalizer_state.npy')
