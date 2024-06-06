import os
import glob
import pickle
import json

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
        if transpose:
            x, y = y, x
            tester.test_data_all_spectra = tester.test_data_all_spectra.T
        pts = np.stack(np.meshgrid(y, x, indexing='ij'), axis=-1).reshape(-1, 2)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))
        zz = scipy.interpolate.griddata(pts, tester.test_data_all_spectra.reshape(-1), (yy, xx), 'linear')
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

    def compare_calculated_percentages(self, result_folders, labels,
                                       output_filename='comparison_calculated_percentages.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.8))
        for i, folder in enumerate(result_folders):
            table = pd.read_csv(os.path.join(folder, 'phase_transition_percentages.csv'))
            ax.plot(table['indices'], table['percentages_estimated'] * 100, linestyle=self.style_list[i], label=labels[i])
        table = pd.read_csv(os.path.join(result_folders[0], 'phase_transition_percentages.csv'))
        ax.plot(table['indices'], table['percentages_true'] * 100, linestyle='--', color='gray', label='Ground truth')
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('Phase transition percentage (%)')
        ax.grid()
        ax.legend()
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
    analyzer.compare_calculated_percentages(folders, labels, output_filename='LTO_50C_comparison_calculated_percentages.pdf')
