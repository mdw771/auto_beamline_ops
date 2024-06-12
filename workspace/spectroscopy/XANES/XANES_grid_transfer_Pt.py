import os
import glob
import pickle
import re
import logging
import json

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
from autobl.steering.io_util import *
from autobl.util import *

from XANES_grid_transfer_LTO import LTOGridTransferTester

torch.set_default_device('cpu')

set_random_seed(123)


class PtGridTransferTester(LTOGridTransferTester):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_data_from_nor(path):
        table = read_nor(path)
        data = table['norm'].to_numpy()
        energies = table['e'].to_numpy()
        _, unique_inds = np.unique(energies, return_index=True)
        unique_inds = np.sort(unique_inds)
        data = data[unique_inds]
        energies = energies[unique_inds]
        return data, energies

    @staticmethod
    def load_all_data(folder):
        flist = glob.glob(os.path.join(folder, '*.nor'))
        flist.sort()
        data_all_spectra = []
        energies = None

        for f in flist:
            d, energies = PtGridTransferTester.load_data_from_nor(f)
            data_all_spectra.append(d)
        data_all_spectra = np.array(data_all_spectra)
        energies = torch.tensor(energies)
        return data_all_spectra, energies

    def load_data(self):
        data_all_spectra, self.test_energies = self.load_all_data(self.test_data_path)
        # Use the first and last spectra as references and the rest as test set
        self.test_data_all_spectra = data_all_spectra[1:-1]
        self.ref_data_all_spectra = data_all_spectra[np.array([0, -1])]
        self.ref_spectra_x = self.test_energies

        ref_spectra_0 = torch.tensor(self.ref_data_all_spectra[0])
        ref_spectra_1 = torch.tensor(self.ref_data_all_spectra[-1])
        self.ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)

        if self.save_plots:
            self._plot_data()

    def get_generic_config(self):
        configs = XANESExperimentGuideConfig(
            dim_measurement_space=1,
            num_candidates=1,
            model_class=botorch.models.SingleTaskGP,
            model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
            noise_variance=1e-6,
            # override_kernel_lengthscale=30,
            lower_bounds=torch.tensor([self.test_energies[0]]),
            upper_bounds=torch.tensor([self.test_energies[-1]]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_params={'gradient_order': 2,
                                         'differentiation_method': 'numerical',
                                         'reference_spectra_x': self.ref_spectra_x,
                                         'reference_spectra_y': self.ref_spectra_y,
                                         'phi_r': 1e2,
                                         'phi_g': 2e-3,  # 2e-2,
                                         'phi_g2': 2e-3,  # 3e-4
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
                params={'threshold': 0.02}
            ),
            use_spline_interpolation_for_posterior_mean=True
        )
        return configs

    def run_acquisition_for_spectrum_index(self, ind, name_prefix='Pt', dataset='test'):
        return super().run_acquisition_for_spectrum_index(ind, name_prefix=name_prefix, dataset=dataset)



if __name__ == '__main__':
    tester = PtGridTransferTester(
        test_data_path='data/raw/Pt-XANES',
        ref_spectra_data_path='data/raw/Pt-XANES',
        output_dir='outputs/grid_transfer_Pt/grid_redoForEach/Pt',
        grid_generation_method='redo_for_each',
        n_initial_measurements=20, n_target_measurements=60, initialization_method='random'
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = PtGridTransferTester(
        test_data_path='data/raw/Pt-XANES',
        ref_spectra_data_path='data/raw/Pt-XANES',
        output_dir='outputs/grid_transfer_Pt/grid_initOfSelf/Pt',
        grid_generation_method='init',
        n_initial_measurements=20, n_target_measurements=60, initialization_method='random'
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = PtGridTransferTester(
        test_data_path='data/raw/Pt-XANES',
        ref_spectra_data_path='data/raw/Pt-XANES',
        output_dir='outputs/grid_transfer_Pt/grid_selectedRef/Pt',
        grid_generation_method='ref',
        grid_generation_spectra_indices=(0, 1),
        grid_intersect_tol=3.0,
        n_initial_measurements=20, n_target_measurements=60, initialization_method='random'
    )
    tester.build()
    tester.run()
    tester.post_analyze()
