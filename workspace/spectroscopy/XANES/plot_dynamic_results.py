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
import sklearn.neighbors
from torch import normal

from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from plot_results import ResultAnalyzer, interpolate_data_on_grid
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
        matplotlib.rcParams['text.usetex'] = True

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
            ax.invert_yaxis()
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
        
    @staticmethod    
    def fit_maximum(x, y, window_size=5):
        window_rad = window_size // 2
        max_pos_int = np.argmax(y)
        ys = y[max_pos_int - window_rad:max_pos_int - window_rad + window_size]
        xs = x[max_pos_int - window_rad:max_pos_int - window_rad + window_size]
        a, b, c = np.polyfit(xs, ys, 2)
        max_pos = - b / (2 * a)
        max_val = a * max_pos ** 2 + b * max_pos + c
        return max_pos, max_val

    def compare_maxima_across_time(self, result_folders, labels, x_data=None, x_label=None, normalizer=None, fit_normalizer_with_true_data=True, 
                                   output_filename='comparison_maxima_across_time.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        for i, folder in enumerate(result_folders):
            true_max_pos_list = []
            true_max_deriv_pos_list = []
            max_pos_list = []
            max_deriv_pos_list = []
            metadata = self.get_metadata(folder)
            tester = self.tester_class(**metadata)
            tester = self.tester_class(**metadata)
            tester.load_data(normalizer=normalizer)
            flist = glob.glob(os.path.join(folder, 'estimated_data_ind_*.csv'))
            flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
            pkl_flist = glob.glob(os.path.join(folder, '*intermediate_data.pkl'))
            pkl_flist = np.array(pkl_flist)[np.argsort([int(re.findall('\d+', x)[0]) for x in pkl_flist])]
            for i_file, (f, pkl_f) in enumerate(zip(flist, pkl_flist)):
                table_spectrum = pd.read_csv(f, index_col=None)
                energies = table_spectrum['energy'].to_numpy()
                estimated_spectrum = table_spectrum['estimated_data'].to_numpy()
                true_spectrum = table_spectrum['true_data'].to_numpy()
                
                with open(pkl_f, 'rb') as pkl_file:
                    intermediate_data = pickle.load(pkl_file)
                energies = intermediate_data['x_dense_list']
                estimated_spectrum = intermediate_data['mu_dense_list'][-1]
                _, true_spectrum = interpolate_data_on_grid(table_spectrum['energy'].to_numpy(), true_spectrum, len(energies))
                
                if normalizer is not None:
                    if fit_normalizer_with_true_data:
                        normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[i])
                    else:
                        normalizer.fit(energies, estimated_spectrum)
                    estimated_spectrum = normalizer.apply(energies, estimated_spectrum)
                    true_spectrum = normalizer.apply(energies, true_spectrum)
                max_pos, max_val = self.fit_maximum(energies, estimated_spectrum, window_size=9)
                true_max_pos, true_max_val = self.fit_maximum(energies, true_spectrum, window_size=9)
                deriv_energies, estimated_deriv = estimate_sparse_gradient(energies, estimated_spectrum)
                deriv_energies, true_deriv = estimate_sparse_gradient(energies, true_spectrum)
                max_deriv_pos, max_deriv_val = self.fit_maximum(deriv_energies, estimated_deriv, window_size=21)
                true_max_deriv_pos, true_max_deriv_val = self.fit_maximum(deriv_energies, true_deriv, window_size=21)
                max_pos_list.append(max_pos)
                true_max_pos_list.append(true_max_pos)
                max_deriv_pos_list.append(max_deriv_pos)
                true_max_deriv_pos_list.append(true_max_deriv_pos)
            if i == 0:
                if x_data is None:
                    x_data = np.arange(len(true_max_pos_list))
                ax.plot(x_data, true_max_pos_list, color='gray', linestyle='--', label='Ground truth')
                ax2 = ax.twinx()
                ax2.plot(x_data, true_max_deriv_pos_list, color='gray', linestyle='--', label='Ground truth', linewidth=0.5)
            ax.plot(x_data, max_pos_list, linestyle=self.style_list[i], label=labels[i])
            ax2.plot(x_data, max_deriv_pos_list, linestyle=self.style_list[i], label=labels[i], linewidth=0.5)
        ax.set_xlabel(x_label if x_label is not None else 'Spectrum index')
        ax.set_ylabel('Energy of maximum\nabsorption (eV)')
        ax2.set_ylabel('Energy of maximum\nderivative of absorption (eV)')
        ax.grid()
        if len(result_folders) > 1:
            ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))
        print("Max error of maximum position: {}".format(np.max(np.abs(np.array(true_max_pos_list) - np.array(max_pos_list)))))
        print("Max error of maximum derivative position: {}".format(np.max(np.abs(np.array(true_max_deriv_pos_list) - np.array(max_deriv_pos_list)))))

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

    def calculate_phase_transition_percentages(self, folder, normalizer=None):
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
            energies = table_spectrum['energy'].to_numpy()
            estimated_spectrum = table_spectrum['estimated_data'].to_numpy()
            true_spectrum = table_spectrum['true_data'].to_numpy()
            if normalizer is not None:
                normalizer.fit(energies, estimated_spectrum)
                estimated_spectrum = normalizer.apply(energies, estimated_spectrum)
                normalizer.fit(energies, true_spectrum)
                true_spectrum = normalizer.apply(energies, true_spectrum)
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
                                       read_precalculated_percentage_data=True, normalizer=None, plot_truth=True,
                                       plot_func="plot", legend=True,
                                       output_filename='comparison_calculated_percentages.pdf',
                                       figsize=(6, 2.8)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
            y_data = np.clip(table['percentages_estimated'] * 100, 0, 100)
            if plot_func == "plot":
                ax.plot(x_data, y_data, linestyle=self.style_list[i], label=labels[i])
            else:
                ax.scatter(x_data, y_data, label=labels[i])
                p = np.polyfit(x_data, y_data, 1)
                y_fit = p[0] * x_data + p[1]
                ax.plot(x_data, y_fit, label="Regression line", linestyle='--', color='gray')
        if plot_truth:
            ax.plot(x_data, np.clip(table['percentages_true'] * 100, 0, 100), linestyle='--', color='gray', label='Ground truth')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Phase transition percentage (\%)')
        ax.grid()
        if legend:
            ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))
        
        max_error = np.max(np.abs(table['percentages_estimated'] - table['percentages_true']))
        max_error_index = np.argmax(np.abs(table['percentages_estimated'] - table['percentages_true']))
        max_error_percentage = max_error / table['percentages_true'][max_error_index] * 100
        print(f'Max error: {max_error:.3f} ({max_error_percentage:.3f}%) at index {max_error_index}')

    def plot_spectrum(self, result_folders, labels, spectrum_index, normalizer=None, output_filename="spectrum.pdf"):
        if normalizer is not None:
            metadata = self.get_metadata(result_folders[0])
            tester = self.tester_class(**metadata)
            tester.load_data(normalizer=normalizer)
            normalizer.fit(tester.test_energies_raw, tester.test_data_all_spectra_raw[spectrum_index])
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        for i, (result_folder, label) in enumerate(zip(result_folders, labels)):
            table = pd.read_csv(os.path.join(result_folder, 'estimated_data_ind_{}.csv'.format(spectrum_index)))
            pkl_file = glob.glob(os.path.join(result_folder, '*index_{}*intermediate_data.pkl'.format(spectrum_index)))[0]
            pkl_file = pickle.load(open(pkl_file, "rb"))
            energies = table['energy']
            estimated_data = table['estimated_data']
            true_data = table['true_data']
            
            # energies = pkl_file['x_dense_list']
            # estimated_data = pkl_file['mu_dense_list'][-1]
            # _, true_data = interpolate_data_on_grid(table['energy'].to_numpy(), true_data.to_numpy(), len(energies))
            
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
        
    def plot_density_estimation(self, points, new_points, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if new_points.ndim == 1:
            new_points = new_points.reshape(-1, 1)
        engine = sklearn.neighbors.KernelDensity(bandwidth=5, kernel='gaussian')
        engine.fit(points)
        density = engine.score_samples(new_points).reshape(-1)
        density = np.exp(density)
        density /= density.max()
        # ax.plot(new_points.reshape(-1), density, color='cyan')
        ax.fill_between(new_points.reshape(-1), density, alpha=0.5)
        return ax
        
    def plot_all_spectra(self, result_folder, indices, labels, normalizer=None, output_filename="all_spectra.pdf", 
                         fit_normalizer_with_true_data=True, plot_func="plot", xtick_interval=None, value_range=(0, 1),
                         alpha=1,
                         plot_measured_data=False, plot_density_estimation=False,
                         plot_figsize=(8, 5), imshow_figsize=(6, 3.5), xlim=None, ylim=None, legend=True, axis_labels=True,
                         linewidth=0.5):
        metadata = self.get_metadata(result_folder)
        tester = self.tester_class(**metadata)
        tester.load_data(normalizer=normalizer)
        
        if plot_func == "plot":
            fig, ax = plt.subplots(1, 1, figsize=plot_figsize)
            cmap_list = matplotlib.colormaps['jet'](np.linspace(0, 1, len(indices)))
        else:
            fig, ax = plt.subplots(1, 1, figsize=imshow_figsize)
            data = []
        all_measured_energies = []
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
                if plot_measured_data:
                    table_meas = pd.read_csv(os.path.join(result_folder, 'measured_data_ind_{}.csv'.format(i)))
                    measured_energies = table_meas['x_measured'].to_numpy()
                    measured_data = table_meas['y_measured'].to_numpy()
                    all_measured_energies.append(measured_energies)
                    if normalizer is not None:
                        measured_data = normalizer.apply(measured_energies, measured_data)
                    ax.scatter(measured_energies, measured_data, s=3, color=cmap_list[i], label=lab)
                x = energies
                y = estimated_data
                ax.plot(x, y, label=lab if not plot_measured_data else None, linewidth=linewidth, color=cmap_list[i], alpha=alpha)
            else:
                x = np.linspace(energies_0[0], energies_0[-1], len(energies_0))
                y = scipy.interpolate.griddata(energies.to_numpy().reshape(-1, 1), estimated_data, x.reshape(-1, 1), method='linear')[:, 0]
                data.append(y)
        if plot_func == "plot":
            if plot_density_estimation:
                all_measured_energies = np.concatenate(all_measured_energies)
                ax = self.plot_density_estimation(all_measured_energies, np.linspace(energies.min(), energies.max(), 100), ax=ax)

            ax.grid()
            ax.tick_params(axis='both', which='major', labelsize=14)
            if legend:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
            if axis_labels:
                ax.set_xlabel('Energy (eV)', fontsize=16)
                ax.set_ylabel('Normalized\nx-ray absorption', fontsize=16)
        else:
            data = np.stack(data).transpose()
            im = ax.imshow(data, extent=[0, data.shape[1], energies_0.max(), energies_0.min()], vmin=value_range[0], vmax=value_range[1],
                           cmap='jet')
            if xtick_interval is not None:
                ax.set_xticks(np.arange(x.min(), x.max(), xtick_interval))
            plt.colorbar(im, fraction=0.046, pad=0.04)
            if axis_labels:
                ax.set_ylabel('Energy (eV)', fontsize=16)
                ax.set_xlabel('Spectrum index', fontsize=16)
            ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, output_filename))
