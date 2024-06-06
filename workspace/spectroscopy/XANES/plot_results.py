import os
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

from autobl.util import *


class ResultAnalyzer:

    style_list = ['-', '-.', ':', (0, (3, 5, 1, 5, 1, 5))]

    def __init__(self, output_dir='factory'):
        self.output_dir = output_dir

        matplotlib.rc('font', family='Times New Roman')
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['pdf.fonttype'] = 42

    def compare_convergence(self, file_list, labels, ref_line_y=0.005, output_filename='comparison_convergence.pdf'):
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
            ax.plot(n_pts_all_files[i], rms_all_files[i], linestyle=self.style_list[i % len(self.style_list)], label=labels[i])
        if ref_line_y is not None:
            orig_xlim, orig_ylim = ax.get_xlim(), ax.get_ylim()
            ax.hlines(ref_line_y, orig_xlim[0], orig_xlim[1], linestyles='--', colors='gray', alpha=0.5)
            ax.text(orig_xlim[0] + (orig_xlim[1] - orig_xlim[0]) * 0.05,
                    ref_line_y + (orig_ylim[1] - orig_ylim[0]) * 0.01,
                    str(ref_line_y), color='gray')
            ax.set_xlim(orig_xlim)
            ax.set_ylim(orig_ylim)
        ax.set_xlabel('Points measured')
        ax.set_ylabel('RMS')
        plt.tight_layout()
        ax.legend(loc='upper right', frameon=True, ncol=1, fontsize=16)
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')

    def plot_intermediate(self, filename, n_cols=3, interval=5, output_filename='intermediate.pdf'):
        data = pickle.load(open(filename, 'rb'))
        n = len(data['mu_list'])
        n_plots = n // interval + 1
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
            ax[i_row][i_col].plot(data['data_x'], data['data_y'], color='gray', alpha=0.5, linestyle='--', label='Ground truth')
            ax[i_row][i_col].set_title('{} points'.format(data['n_measured_list'][iter]))
            ax[i_row][i_col].set_xlabel('Energy (eV)')
            ax[i_row][i_col].grid(True)
            if i == 0:
                ax[i_row][i_col].legend(frameon=False)
            i_plot += 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')

    def compare_estimates(self, file_list, labels, at_n_pts=20,
                          zoom_in_range_x=None, zoom_in_range_y=None,
                          output_filename='comparison_estimate.pdf'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        data = pickle.load(open(file_list[0], 'rb'))
        true_x, true_y = data['data_x'], data['data_y']

        def plot_ax(ax, tick_interval=None):
            for i, f in enumerate(file_list):
                data = pickle.load(open(f, 'rb'))
                at_iter = data['n_measured_list'].index(at_n_pts)
                x, y = data['data_x'], data['mu_list'][at_iter]
                ax.plot(x, y, linewidth=1, linestyle=self.style_list[i % len(self.style_list)], label=labels[i])
                if i == 0:
                    ax.scatter(data['measured_x_list'][at_iter], data['measured_y_list'][at_iter], s=4)
            ax.plot(true_x, true_y, color='gray', alpha=0.5, linestyle='--', label='Ground truth')
            if tick_interval is not None:
                ax.set_xticks(np.arange(np.ceil(true_x.min() / tick_interval) * tick_interval, true_x.max(), tick_interval))
            ax.legend()
            ax.set_xlabel('Energy (eV)')
            ax.grid()

        plot_ax(ax, tick_interval=20)
        if zoom_in_range_x is not None:
            fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(5, 5))
            plot_ax(ax_zoom)
            ax_zoom.get_legend().set_visible(False)
            ax_zoom.set_xlim(zoom_in_range_x)
            ax_zoom.set_ylim(zoom_in_range_y)
            ax_zoom.tick_params(axis='x', labelsize=18)
            ax_zoom.tick_params(axis='y', labelsize=18)
            ax_zoom.set_xlabel('')
            ax.add_patch(patches.Rectangle((zoom_in_range_x[0], zoom_in_range_y[0]),
                                                zoom_in_range_x[1] - zoom_in_range_x[0],
                                                zoom_in_range_y[1] - zoom_in_range_y[0],
                                                linestyle='--', edgecolor='orange', facecolor='none'))
            fig_zoom.savefig(os.path.join(self.output_dir, os.path.splitext(output_filename)[0] + '_zoom.pdf'),
                             bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, output_filename), bbox_inches='tight')


if __name__ == '__main__':
    # flist = [glob.glob('outputs/random_init/Sample1_50C_XANES_*_intermediate_data.pkl')[0],
    #          glob.glob('outputs/random_init_no_reweighting/Sample1_50C_XANES_*_intermediate_data.pkl')[0],
    #          glob.glob('outputs/random_init_posterior_stddev/Sample1_50C_XANES_*_intermediate_data.pkl')[0]
    #          ]
    flist = [glob.glob('outputs/random_init/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/random_init_no_reweighting/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/random_init_posterior_stddev/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/random_init_uniform_sampling/YBCO3data_*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              'Posterior uncertainty only',
              'Uniform sampling'
              ]
    analyzer = ResultAnalyzer(output_dir='factory')
    analyzer.compare_convergence(flist, labels, output_filename='YBCO_comparison_convergence.pdf')
    analyzer.plot_intermediate(flist[0], interval=6, output_filename='YBCO_intermediate.pdf')
    analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=32,
                               zoom_in_range_x=(9010, 9030), zoom_in_range_y=(0.7, 1.1),
                               output_filename='YBCO_intermediate_atNPts_32.pdf')
