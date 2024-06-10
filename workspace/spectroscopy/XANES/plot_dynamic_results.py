import os
import glob
import pickle
import json
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import pandas as pd
import scipy.interpolate

from autobl.util import *
from plot_results import ResultAnalyzer
from XANES_grid_transfer_LTO import LTOGridTransferTester


class DynamicExperimentResultAnalyzer:

    style_list = ResultAnalyzer.style_list

    def __init__(self, output_dir='factory'):
        self.output_dir = output_dir

        matplotlib.rc('font', family='Times New Roman')
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['pdf.fonttype'] = 42

    @staticmethod
    def get_metadata(folder):
        with open(os.path.join(folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        return metadata

    def plot_data(self, folder, transpose=False, xtick_interval=None, value_range=(0, 1.7), output_filename='test_data.pdf'):
        metadata = self.get_metadata(folder)
        tester = LTOGridTransferTester(**metadata)
        tester.load_data()

        fig, ax = plt.subplots(figsize=(6, 3.5))
        x, y = to_numpy(tester.test_energies.squeeze()), np.arange(len(tester.test_data_all_spectra))
        all_spectra = tester.test_data_all_spectra.copy()
        if transpose:
            x, y = y, x
            all_spectra = all_spectra.T
        pts = np.stack(np.meshgrid(y, x, indexing='ij'), axis=-1).reshape(-1, 2)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))
        zz = scipy.interpolate.griddata(pts, all_spectra.reshape(-1), (yy, xx), 'linear')
        im = ax.imshow(zz, extent=[x.min(), x.max(), y.max(), y.min()], vmin=value_range[0], vmax=value_range[1],
                       cmap='jet')
        if xtick_interval is not None:
            ax.set_xticks(np.arange(x.min(), x.max(), xtick_interval))
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if transpose:
            ax.set_ylabel('Energy (eV)')
            ax.set_xlabel('Spectrum index')
        else:
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Spectrum index')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

        # Beginning and end spectra
        x, y = to_numpy(tester.test_energies.squeeze()), tester.test_data_all_spectra
        fig, ax = plt.subplots(2, 1, figsize=(4, 4))
        ax[0].plot(x, y[0])
        ax[0].set_ylabel('Normalized absorption')
        ax[0].grid()
        ax[1].plot(x, y[-1])
        ax[1].set_ylabel('Normalized absorption')
        ax[1].set_xlabel('Energy (eV)')
        ax[1].grid()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, os.path.splitext(output_filename)[0] + '_spectra.pdf'))

    def compare_rms_across_time(self, result_folders, labels, output_filename='comparison_rms_across_time.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        for i, folder in enumerate(result_folders):
            rms_list = np.loadtxt(os.path.join(folder, 'rms_all_test_spectra.txt'))
            ax.plot(rms_list, linestyle=self.style_list[i], label=labels[i])
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('RMS')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def calculate_phase_transition_percentages(self, folder):
        metadata = self.get_metadata(folder)
        tester = LTOGridTransferTester(**metadata)
        tester.load_data()
        indices = []
        percentages_estimated = []
        percentages_true = []
        fitting_residue_estimated = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
        for f in flist:
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table_spectrum = pd.read_csv(f, index_col=None)
            estimated_spectrum = table_spectrum['estimated_data'].to_numpy()
            true_spectrum = table_spectrum['true_data'].to_numpy()
            p_estimated, r = tester.get_phase_transition_percentage(estimated_spectrum, return_fitting_residue=True,
                                                                    add_constant_term=True)
            percentages_estimated.append(p_estimated)
            fitting_residue_estimated.append(r)
            p_true, r = tester.get_phase_transition_percentage(true_spectrum, return_fitting_residue=True,
                                                               add_constant_term=True)
            percentages_true.append(p_true)
        table = pd.DataFrame(data={'indices': indices, 'percentages_estimated': percentages_estimated,
                                   'percentages_true': percentages_true})
        return table

    def compare_calculated_percentages(self, result_folders, labels, read_precalculated_percentage_data=True,
                                       output_filename='comparison_calculated_percentages.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.8))
        table = {}
        for i, folder in enumerate(result_folders):
            if read_precalculated_percentage_data:
                table = pd.read_csv(os.path.join(folder, 'phase_transition_percentages.csv'))
            else:
                metadata = self.get_metadata(folder)
                tester = LTOGridTransferTester(**metadata)
                tester.load_data()
                table = tester.calculate_phase_transition_percentages()
            ax.plot(table['indices'], np.clip(table['percentages_estimated'] * 100, 0, 100), linestyle=self.style_list[i], label=labels[i])
        ax.plot(table['indices'], np.clip(table['percentages_true'] * 100, 0, 100), linestyle='--', color='gray', label='Ground truth')
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('Phase transition percentage (%)')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def plot_spectrum(self, result_folders, labels, spectrum_index, output_filename):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        for i, (result_folder, label) in enumerate(zip(result_folders, labels)):
            table = pd.read_csv(os.path.join(result_folder, 'estimated_data_ind_{}.csv'.format(spectrum_index)))
            energies = table['energy']
            estimated_data = table['estimated_data']
            true_data = table['true_data']
            if i == 0:
                ax.plot(energies, true_data, color='gray', linestyle='--', label='Ground truth')
            ax.plot(energies, estimated_data, linestyle=self.style_list[i], label=label)
        ax.grid()
        ax.legend()
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized absorption')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))


if __name__ == '__main__':
    folders = [
        'outputs/grid_transfer/grid_redoForEach/Sample1_50C',
        'outputs/grid_transfer/grid_initOfSelf/Sample1_50C',
        'outputs/grid_transfer/grid_selectedRef/Sample1_50C',
    ]
    labels = [
        'Run for each',
        'Initial spectrum',
        'Selected from ref. experiments'
    ]

    analyzer = DynamicExperimentResultAnalyzer()
    analyzer.plot_data(folders[0], transpose=True, output_filename='LTO_50C_test_data.pdf')
    analyzer.plot_data('outputs/grid_transfer/grid_initOfSelf/Sample3_70C', transpose=True, xtick_interval=5, output_filename='LTO_70C_test_data.pdf')
    analyzer.compare_rms_across_time(folders, labels, output_filename='LTO_50C_comparison_rms_across_time.pdf')
    analyzer.compare_calculated_percentages(folders, labels, read_precalculated_percentage_data=False, output_filename='LTO_50C_comparison_calculated_percentages.pdf')
    analyzer.plot_spectrum(folders, labels, 80, output_filename='LTO_50C_estimated_spectra_80.pdf')
    analyzer.plot_spectrum(folders, labels, 90, output_filename='LTO_50C_estimated_spectra_90.pdf')
