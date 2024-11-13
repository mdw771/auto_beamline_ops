import os
import glob
import pickle
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import pandas as pd
import scipy.interpolate
from torch import normal

from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from plot_results import ResultAnalyzer
from XANES_grid_transfer_LTO import LTOGridTransferTester
from XANES_grid_transfer_Pt import PtGridTransferTester

import logging
logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)


class DynamicExperimentResultAnalyzer:

    style_list = ResultAnalyzer.style_list

    def __init__(self, tester_class=LTOGridTransferTester, output_dir='factory'):
        self.output_dir = output_dir
        self.tester_class = tester_class

        matplotlib.rc('font', family='Times New Roman')
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['pdf.fonttype'] = 42

    @staticmethod
    def get_metadata(folder):
        with open(os.path.join(folder, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        return metadata

    def plot_data(self, folder, transpose=False, xtick_interval=None, value_range=(0, 1.7), normalizer=None, plot_raw_data=False,
                  plot_func='imshow', line_plot_labels=None, output_filename='test_data.pdf'):
        metadata = self.get_metadata(folder)
        tester = self.tester_class(**metadata)
        tester.load_data(normalizer=normalizer)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        if plot_raw_data:
            x, y = to_numpy(tester.test_energies_raw.squeeze()), np.arange(len(tester.test_data_all_spectra))
            all_spectra = tester.test_data_all_spectra_raw.copy()
        else:
            x, y = to_numpy(tester.test_energies.squeeze()), np.arange(len(tester.test_data_all_spectra))
            all_spectra = tester.test_data_all_spectra.copy()

        if plot_func == 'imshow':
            if transpose:
                x, y = y, x
                all_spectra = all_spectra.T
            pts = np.stack(np.meshgrid(y, x, indexing='ij'), axis=-1).reshape(-1, 2)
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))
            zz = scipy.interpolate.griddata(pts, all_spectra.reshape(-1), (yy, xx), 'linear')
            if value_range is None:
                value_range = (None, None)
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
        elif plot_func == 'plot':
            if not plot_raw_data:
                all_spectra = np.concatenate([tester.ref_data_all_spectra[0][None, :], tester.test_data_all_spectra, tester.ref_data_all_spectra[-1][None, :]])
            else:
                all_spectra = np.concatenate([tester.ref_data_all_spectra_raw[0][None, :], tester.test_data_all_spectra_raw, tester.ref_data_all_spectra_raw[-1][None, :]])
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap_list = matplotlib.colormaps['jet'](np.linspace(0, 1, all_spectra.shape[0]))
            for i in range(all_spectra.shape[0]):
                lab = line_plot_labels[i] if line_plot_labels is not None else None
                ax.plot(x, all_spectra[i], label=lab, linewidth=0.5, color=cmap_list[i])
            if line_plot_labels is not None:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Normalized\nx-ray absorption')
            plt.grid()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

        # Beginning and end spectra
        for name, dset in (('spectra', tester.test_data_all_spectra), ('ref_spectra', tester.ref_data_all_spectra)):
            x, y = to_numpy(tester.test_energies.squeeze()), dset
            fig, ax = plt.subplots(2, 1, figsize=(4, 4))
            ax[0].plot(x, y[0])
            ax[0].set_ylabel('Normalized\nx-ray absorption')
            ax[0].grid()
            ax[1].plot(x, y[-1])
            ax[1].set_ylabel('Normalized\nx-ray absorption')
            ax[1].set_xlabel('Energy (eV)')
            ax[1].grid()
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, os.path.splitext(output_filename)[0] + '_{}.pdf'.format(name)))

    def compare_rms_across_time(self, result_folders, labels, output_filename='comparison_rms_across_time.pdf', read_data=True, normalizer=None, fit_normalizer_with_true_data=True):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        for i, folder in enumerate(result_folders):
            if read_data:
                rms_list = np.loadtxt(os.path.join(folder, 'rms_all_test_spectra.txt'))
            else:
                rms_list = []
                metadata = self.get_metadata(folder)
                tester = self.tester_class(**metadata)
                tester.load_data()
                flist = glob.glob(os.path.join(folder, 'estimated_data_ind_*.csv'))
                flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
                for f in flist:
                    table_spectrum = pd.read_csv(f, index_col=None)
                    energies = table_spectrum['energy']
                    estimated_spectrum = table_spectrum['estimated_data'].to_numpy()
                    true_spectrum = table_spectrum['true_data'].to_numpy()
                    if normalizer is not None:
                        if fit_normalizer_with_true_data:
                            normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[i])
                        else:
                            normalizer.fit(energies, estimated_spectrum)
                        estimated_spectrum = normalizer.apply(energies, estimated_spectrum)
                        true_spectrum = normalizer.apply(energies, true_spectrum)
                    rms_list.append(rms(estimated_spectrum, true_spectrum))
            ax.plot(rms_list, linestyle=self.style_list[i], label=labels[i])
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('RMS')
        ax.set_xticks(np.arange(0, len(rms_list), 20, dtype=int))
        ax.grid()
        if len(result_folders) > 1:
            ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def calculate_phase_transition_percentages(self, folder):
        metadata = self.get_metadata(folder)
        tester = self.tester_class(**metadata)
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

    def compare_calculated_percentages(self, result_folders, labels, x_data=None, x_label='Spectrum index', 
                                       read_precalculated_percentage_data=True, normalizer=None,
                                       output_filename='comparison_calculated_percentages.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.8))
        table = {}
        for i, folder in enumerate(result_folders):
            if read_precalculated_percentage_data:
                table = pd.read_csv(os.path.join(folder, 'phase_transition_percentages.csv'))
            else:
                metadata = self.get_metadata(folder)
                tester = self.tester_class(**metadata)
                tester.load_data(normalizer=normalizer)
                table = tester.calculate_phase_transition_percentages(normalizer=normalizer)
            if x_data is None:
                x_data = table['indices']
            ax.plot(x_data, np.clip(table['percentages_estimated'] * 100, 0, 100), linestyle=self.style_list[i], label=labels[i])
        ax.plot(x_data, np.clip(table['percentages_true'] * 100, 0, 100), linestyle='--', color='gray', label='Ground truth')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Phase transition percentage (%)')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))

    def plot_spectrum(self, result_folders, labels, spectrum_index, normalizer=None, output_filename="spectrum.pdf"):
        if normalizer is not None:
            metadata = self.get_metadata(result_folders[0])
            tester = self.tester_class(**metadata)
            tester.load_data(normalizer=normalizer)
            normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[spectrum_index])
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        for i, (result_folder, label) in enumerate(zip(result_folders, labels)):
            table = pd.read_csv(os.path.join(result_folder, 'estimated_data_ind_{}.csv'.format(spectrum_index)))
            energies = table['energy']
            estimated_data = table['estimated_data']
            true_data = table['true_data']
            if normalizer is not None:
                estimated_data = normalizer.apply(energies, estimated_data)
                true_data = normalizer.apply(energies, true_data)
            if i == 0:
                ax.plot(energies, true_data, color='gray', linestyle='--', label='Ground truth')
            ax.plot(energies, estimated_data, linestyle=self.style_list[i], label=label)
        ax.grid()
        ax.legend()
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized\nx-ray absorption')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))
        
    def plot_all_spectra(self, result_folder, indices, labels, normalizer=None, output_filename="all_spectra.pdf", 
                         fit_normalizer_with_true_data=True, plot_func="plot", xtick_interval=None, value_range=(0, 1)):
        metadata = self.get_metadata(result_folder)
        tester = self.tester_class(**metadata)
        tester.load_data(normalizer=normalizer)
        
        if plot_func == "plot":
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            cmap_list = matplotlib.colormaps['jet'](np.linspace(0, 1, len(indices)))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            data = []
        for i in indices:
            table = pd.read_csv(os.path.join(result_folder, 'estimated_data_ind_{}.csv'.format(i)))
            energies = table['energy']
            energies_0 = energies.to_numpy()
            estimated_data = table['estimated_data']
            if normalizer is not None:
                if fit_normalizer_with_true_data:
                    normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[i])
                else:
                    normalizer.fit(energies, estimated_data)
                estimated_data = normalizer.apply(energies, estimated_data)
            if plot_func == "plot":
                lab = labels[i]
                ax.plot(energies, estimated_data, label=lab, linewidth=0.5, color=cmap_list[i])
            else:
                x = np.linspace(energies_0[0], energies_0[-1], len(energies_0))
                y = scipy.interpolate.griddata(energies.to_numpy().reshape(-1, 1), estimated_data, x.reshape(-1, 1), method='linear')[:, 0]
                data.append(y)
        if plot_func == "plot":
            ax.grid()
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Normalized\nx-ray absorption')
        else:
            data = np.stack(data).transpose()
            im = ax.imshow(data, extent=[0, data.shape[1], energies_0.max(), energies_0.min()], vmin=value_range[0], vmax=value_range[1],
                           cmap='jet')
            if xtick_interval is not None:
                ax.set_xticks(np.arange(x.min(), x.max(), xtick_interval))
            plt.colorbar(im, fraction=0.046, pad=0.04)
            ax.set_ylabel('Energy (eV)')
            ax.set_xlabel('Spectrum index')
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))
