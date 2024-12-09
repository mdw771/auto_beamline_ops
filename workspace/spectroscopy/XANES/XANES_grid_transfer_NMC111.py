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
import scipy

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.steering.acquisition import *
from autobl.steering.optimization import *
from autobl.steering.experiment import SimulatedScanningExperiment
from autobl.steering.io_util import *
from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools

from XANES_grid_transfer_LTO import LTOGridTransferTester

torch.set_default_device('cpu')


class NMC111GridTransferTester(LTOGridTransferTester):

    lower_bound = 8232
    upper_bound = 8484

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_data_from_csv(path, normalizer=None, crop=True):
        table = pd.read_csv(path)
        data = table['absorption'].to_numpy()
        energies = table['energy'].to_numpy()
        _, unique_inds = np.unique(energies, return_index=True)
        unique_inds = np.sort(unique_inds)
        data = data[unique_inds]
        energies = energies[unique_inds]
        
        if normalizer is not None:
            normalizer.fit(energies, data)
            data = normalizer.apply(energies, data)
        
        if crop:
            # Only keep 8233 - 8483 eV
            mask = (energies >= __class__.lower_bound) & (energies <= __class__.upper_bound)
            data = data[mask]
            energies = energies[mask]
        return data, energies
    
    def build_point_grid(self, spectrum_ind=None):
        return torch.arange(self.lower_bound, self.upper_bound, 0.5)

    @staticmethod
    def load_all_data(folder, filename_pattern, normalizer=None, crop=True):
        flist = glob.glob(os.path.join(folder, filename_pattern))
        flist.sort()
        data_all_spectra = []
        energies = []

        for f in flist:
            d, e = __class__.load_data_from_csv(f, normalizer=normalizer, crop=crop)
            data_all_spectra.append(d)
            energies.append(e)
        return data_all_spectra, energies

    def load_data(self, normalizer=False):
        self.test_data_all_spectra, self.test_energies = __class__.load_all_data(self.test_data_path, filename_pattern=self.test_data_filename_pattern, normalizer=normalizer)
        self.ref_data_all_spectra, self.ref_energies = __class__.load_all_data(self.ref_spectra_data_path, filename_pattern=self.ref_data_filename_pattern, normalizer=normalizer)

    def get_generic_config(self):
        configs = super().get_generic_config()
        configs.lower_bounds = torch.tensor([self.lower_bound])
        configs.upper_bounds = torch.tensor([self.upper_bound])
        configs.use_spline_interpolation_for_posterior_mean = True
        # configs.override_kernel_lengthscale = 20
        return configs

    def get_gp_interpolation_for_spectrum_index(self, ind):
        query_x = torch.arange(self.lower_bound, self.upper_bound, 0.5).reshape(-1, 1)
        data_y = to_tensor(self.test_data_all_spectra[ind]).reshape(-1, 1)
        data_x = to_tensor(self.test_energies[ind]).reshape(-1, 1)
        interpolated_data = self.get_gp_interpolation(data_x, data_y, query_x=query_x)
        interpolated_data = to_numpy(interpolated_data.squeeze())
        if self.save_plots:
            self.save_spectrum_estimate_plot_and_data(ind, to_numpy(query_x.squeeze()), interpolated_data,
                                                      x_measured=to_numpy(data_x.squeeze()),
                                                      y_measured=to_numpy(data_y.squeeze()))
        return interpolated_data
    
    def run(self):
        if self.grid_generation_method != 'redo_for_each':
            self.build_point_grid()
        for ind in range(len(self.test_data_all_spectra)):
            if self.grid_generation_method == 'redo_for_each':
                self.build_point_grid(spectrum_ind=ind)
            self.get_gp_interpolation_for_spectrum_index(ind)
            
    def calculate_phase_transition_percentages(self, normalizer=None, normalize_percentages=False):
        indices = []
        percentages_estimated = []
        percentages_true = []
        fitting_residue_estimated = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
        
        # Normalize and detilt ref spectra
        if normalizer is not None:
            ref_spectra_normalized = []
            for i in range(len(self.ref_data_all_spectra)):
                normalizer.fit(self.ref_energies[i], self.ref_data_all_spectra[i])
                ref_spectra_normalized.append(normalizer.apply(self.ref_energies[i], self.ref_data_all_spectra[i]))

        for i, f in enumerate(flist):
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table = pd.read_csv(f, index_col=None)
            energies = table['energy'].to_numpy()
            estimated_spectrum = table['estimated_data'].to_numpy()
            true_spectrum = table['true_data'].to_numpy()
            
            mask = (energies > self.ref_energies[0][0]) & (energies < self.ref_energies[0][-1])
            energies = energies[mask]
            estimated_spectrum = estimated_spectrum[mask]
            true_spectrum = true_spectrum[mask]
            
            # Interpolate ref spectrum
            ref_spectra_normalized_interpolated = []
            for i in range(len(ref_spectra_normalized)):
                ref_spectra_normalized_interpolated.append(
                    np.squeeze(scipy.interpolate.griddata(self.ref_energies[i].reshape(-1, 1), ref_spectra_normalized[i], energies.reshape(-1, 1), 'linear'))
                )
            ref_spectra_normalized_interpolated = np.stack(ref_spectra_normalized_interpolated)
            
            # Normalize and detilt if normalizer is provided
            if normalizer is not None:
                normalizer.fit(energies, estimated_spectrum)
                estimated_spectrum = normalizer.apply(energies, estimated_spectrum)
                true_spectrum = normalizer.apply(energies, true_spectrum)
            
            p_estimated, r = self.get_phase_transition_percentage(estimated_spectrum, ref_spectra_normalized_interpolated,
                                                                  return_fitting_residue=True)
            percentages_estimated.append(p_estimated)
            fitting_residue_estimated.append(r)
            p_true = self.get_phase_transition_percentage(true_spectrum, ref_spectra_normalized_interpolated)
            percentages_true.append(p_true)
            
        if normalize_percentages:
            percentages_estimated = np.array(percentages_estimated) / np.max(percentages_estimated)
            percentages_true = np.array(percentages_true) / np.max(percentages_true)

        table = pd.DataFrame(data={'indices': indices,
                                   'percentages_estimated': percentages_estimated,
                                   'percentages_true': percentages_true,
                                   'fitting_residue_estimated': fitting_residue_estimated
                                   })
        return table



if __name__ == '__main__':
    normalizer = xanestools.XANESNormalizer(fit_ranges=((8234, 8312), (8385, 8481)), edge_loc=8343, normalization_order=1)
    
    set_random_seed(126)
    tester = NMC111GridTransferTester(
        test_data_path='data/raw/NMC111',
        test_data_filename_pattern="Adaptive*",
        ref_spectra_data_path='data/raw/NMC111',
        ref_data_filename_pattern="Reference*",
        output_dir='outputs/grid_transfer_NMC111/grid_redoForEach/NMC111',
    )
    tester.build()
    tester.run()
    tester.post_analyze(normalizer=normalizer, normalize_percentages=True)
