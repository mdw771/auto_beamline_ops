import os
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from autobl.util import *


class ResultAnalyzer:

    def __init__(self, output_dir='factory'):
        self.output_dir = output_dir

        matplotlib.rc('font', family='Times New Roman')
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['pdf.fonttype'] = 42

    def compare_convergence(self, file_list, labels, output_filename='comparison_convergence.pdf'):
        rms_all_files = []
        n_pts_all_files = []
        for f in file_list:
            rms_list = []
            n_pts = []
            data = pickle.load(open(f, 'rb'))
            data_true = data['data_y']
            for i, data_estimated in enumerate(data['mu_list']):
                r = rms(data_estimated, data_true)
                rms_list.append(r)
                n_pts.append(data['n_measured_list'][i])
            rms_all_files.append(rms_list)
            n_pts_all_files.append(n_pts)

        fig, ax = plt.subplots(1, 1)
        for i in range(len(rms_all_files)):
            ax.plot(n_pts_all_files[i], rms_all_files[i], label=labels[i])
        ax.set_xlabel('Points measured')
        ax.set_ylabel('RMS')
        plt.tight_layout()
        ax.legend(loc='upper right', frameon=True, ncol=1)
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')

    def plot_intermediate(self, filename, n_cols=3, interval=5, output_filename='intermediate.pdf'):
        data = pickle.load(open(filename, 'rb'))
        n = len(data['mu_list'])
        n_plots = n // interval
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        i_plot = 0
        for i, iter in enumerate(range(0, n, interval)):
            i_col = i_plot % n_cols
            i_row = i_plot // n_cols
            ax[i_row][i_col].plot(data['data_x'], data['mu_list'][iter], linewidth=1, label='Estimated')
            ax[i_row][i_col].fill_between(data['data_x'],
                                          data['mu_list'][iter] - data['sigma_list'][iter],
                                          data['mu_list'][iter] + data['sigma_list'][iter], alpha=0.5)
            ax[i_row][i_col].scatter(data['measured_x_list'][iter], data['measured_y_list'][iter], s=4, label='Measured')
            ax[i_row][i_col].plot(data['data_x'], data['data_y'], color='gray', alpha=0.5, label='Ground truth')
            ax[i_row][i_col].set_title('{} points'.format(data['n_measured_list'][iter]))
            ax[i_row][i_col].set_xlabel('Energy (eV)')
            ax[i_row][i_col].grid(True)
            if i == 0:
                ax[i_row][i_col].legend(frameon=False)
            i_plot += 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')

    def compare_estimates(self, file_list, labels, at_n_pts=20, output_filename='comparison_estimate.pdf'):
        fig, ax = plt.subplots(1, 1)
        data = pickle.load(open(file_list[0], 'rb'))
        true_x, true_y = data['data_x'], data['data_y']
        ax.plot(true_x, true_y, color='gray', alpha=0.5, label='Ground truth')
        for i, f in enumerate(file_list):
            data = pickle.load(open(f, 'rb'))
            at_iter = data['n_measured_list'].index(at_n_pts)
            x, y = data['data_x'], data['mu_list'][at_iter]
            ax.plot(x, y, linewidth=1, label=labels[i])
        ax.legend()
        ax.set_xlabel('Energy (eV)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')


if __name__ == '__main__':
    # flist = [glob.glob('outputs/random_init/Sample1_50C_XANES_*_intermediate_data.pkl')[0],
    #          glob.glob('outputs/random_init_no_reweighting/Sample1_50C_XANES_*_intermediate_data.pkl')[0],
    #          glob.glob('outputs/random_init_posterior_stddev/Sample1_50C_XANES_*_intermediate_data.pkl')[0]
    #          ]
    flist = [glob.glob('outputs/random_init/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/random_init_no_reweighting/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/random_init_posterior_stddev/YBCO3data_*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acquisition + acquisition reweighting',
              'Comprehensive acquisition',
              'Posterior standard deviation only']
    analyzer = ResultAnalyzer(output_dir='factory')
    analyzer.compare_convergence(flist, labels, output_filename='YBCO_comparison_convergence.pdf')
    analyzer.plot_intermediate(flist[0], output_filename='YBCO_intermediate.pdf')
    analyzer.compare_estimates(flist[0::2], labels[0::2], at_n_pts=30, output_filename='YBCO_intermediate_atNPts_30.pdf')
