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
from autobl.steering.model import *
from autobl.util import *
from autobl.steering.guide import *
from autobl.steering.io_util import LTORawDataset
import autobl.tools.spectroscopy.xanes as xanestools

torch.set_default_device('cpu')

set_random_seed(123)

dataset = LTORawDataset('data/raw/LiTiO_XANES/rawdata', filename_pattern="LTOsample2.[0-9]*")
data_all_spectra = dataset.data
energies = dataset.energies_ev
data = data_all_spectra[len(data_all_spectra) // 2]

# plt.figure()
# plt.plot(energies, data)
# plt.show()

normalizer = xanestools.XANESNormalizer()
normalizer.fit(energies, data, fit_ranges=((4900, 4950), (5100, 5200)))

mask = (energies >= 4936) & (energies <= 5006)
data_all_spectra = data_all_spectra[:, mask]
data = data_all_spectra[len(data_all_spectra) // 2]
energies = energies[mask]
energies = torch.tensor(energies)
ref_spectra_0 = torch.tensor(data_all_spectra[0])
ref_spectra_1 = torch.tensor(data_all_spectra[-1])
ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
ref_spectra_x = energies

model_params = {
    "dim_recon_spec": 1000,
    "dim_spec_encoded": 256,
    "dim_feat_encoded": 256,
    "add_pooling": False
}

configs = SLDAS1DExperimentGuideConfig(
    dim_measurement_space=1,
    num_candidates=1,
    model_class=ConvMLPModel,
    model_params=model_params,
    model_path="trained_models/model_convMLP_dimReconSpec1000_dimSpecEncoded256_dimFeatEncoded256_featPos3NNs_gaussRandEnc_lr1e-3/best_model.pth",
    lower_bounds=torch.tensor([energies[0]]),
    upper_bounds=torch.tensor([energies[-1]]),
    n_eval_pixels=1000
)

analyzer_configs = ExperimentAnalyzerConfig(
    name='Sample3_70C_XANES',
    output_dir='outputs/LTO_raw_randInit',
    n_plot_interval=5
)

normalizer.save_state('outputs/LTO_raw_randInit/normalizer_state.npy')

experiment = SimulatedScanningExperiment(configs, guide_class=SLDAS1DExperimentGuide, run_analysis=True, analyzer_configs=analyzer_configs)
experiment.build(energies, data)
experiment.run(n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='random')
