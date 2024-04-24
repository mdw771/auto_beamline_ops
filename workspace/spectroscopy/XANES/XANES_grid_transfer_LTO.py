import os
import glob
import pickle
import re

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
from autobl.util import *

torch.set_default_device('cpu')

set_random_seed(123)


class LTOGridTransferTester:

    def __init__(self, test_data_path, ref_spectra_data_path, output_dir='outputs'):
        self.test_data_path = test_data_path
        self.ref_spectra_data_path = ref_spectra_data_path
        self.test_data_all_spectra = None
        self.test_energies = None
        self.ref_spectra_x = None
        self.ref_spectra_y = None
        self.output_dir = output_dir
        self.results = {'spectrum_index': [], 'rms': []}
        self.point_grid = None
        self.save_plots = True
        self.debug = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def load_data_from_csv(path):
        table = pd.read_csv(path, header=None)
        data_all_spectra = table.iloc[1:].to_numpy()
        energies = torch.tensor(table.iloc[0].to_numpy())
        return data_all_spectra, energies

    def load_data(self):
        self.test_data_all_spectra, self.test_energies = self.load_data_from_csv(self.test_data_path)

        ref_data_all_spectra, self.ref_spectra_x = self.load_data_from_csv(self.ref_spectra_data_path)
        ref_spectra_0 = torch.tensor(ref_data_all_spectra[0])
        ref_spectra_1 = torch.tensor(ref_data_all_spectra[-1])
        self.ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)

        if self.save_plots:
            self._plot_data()

    def _plot_data(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x, y = np.meshgrid(to_numpy(self.test_energies.squeeze()),
                           np.arange(len(self.test_data_all_spectra)))
        ax.view_init(elev=20, azim=-100)
        ax.plot_surface(x, y, self.test_data_all_spectra, cmap='jet')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Spectrum index')
        fig.savefig(os.path.join(self.output_dir, 'test_data_3d.pdf'))
        plt.close(fig)

        fig, ax = plt.subplots()
        x, y = to_numpy(self.test_energies.squeeze()), np.arange(len(self.test_data_all_spectra))
        pts = np.stack(np.meshgrid(y, x, indexing='ij'), axis=-1).reshape(-1, 2)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))
        zz = scipy.interpolate.griddata(pts, self.test_data_all_spectra.reshape(-1), (yy, xx), 'linear')
        im = ax.imshow(zz, extent=[x.min(), x.max(), y.max(), y.min()], cmap='jet')
        plt.colorbar(im)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Spectrum index')
        fig.savefig(os.path.join(self.output_dir, 'test_data.pdf'))
        plt.close(fig)

    def get_generic_config(self):
        configs = XANESExperimentGuideConfig(
            dim_measurement_space=1,
            num_candidates=1,
            model_class=botorch.models.SingleTaskGP,
            model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
            override_kernel_lengthscale=7,
            noise_variance=1e-6,
            lower_bounds=torch.tensor([self.test_energies[0]]),
            upper_bounds=torch.tensor([self.test_energies[-1]]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_params={'gradient_order': 2,
                                         'differentiation_method': 'numerical',
                                         'reference_spectra_x': self.ref_spectra_x,
                                         'reference_spectra_y': self.ref_spectra_y,
                                         'phi_r': None,
                                         'phi_g': None,
                                         'phi_g2': None,
                                         'beta': 0.999,
                                         'gamma': 0.95,
                                         'addon_term_lower_bound': 3e-2,
                                         'debug': False},
            n_updates_create_acqf_weight_func=5,
            acqf_weight_func_floor_value=0.01,
            acqf_weight_func_post_edge_gain=3.0,
            optimizer_class=DiscreteOptimizer,
            optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                              'optim_func_params': {
                                  'choices': torch.linspace(0, 1, 1000)[:, None]
                              }
                              },
            stopping_criterion_configs=None,
        )
        return configs

    def run_acquisition_for_spectrum_index(self, ind):
        configs = self.get_generic_config()

        analyzer_configs = ExperimentAnalyzerConfig(
            name='LTO_50C_index_{}'.format(ind),
            output_dir=self.output_dir,
            n_plot_interval=5,
            save=self.save_plots
        )

        data = self.test_data_all_spectra[ind]
        experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(self.test_energies, data)
        experiment.run(n_initial_measurements=10, n_target_measurements=40)
        if self.save_plots:
            interpolated_data = to_numpy(experiment.guide.get_posterior_mean_and_std(self.test_energies)[0].squeeze())
            self.save_spectrum_estimate_plot_and_data(ind, to_numpy(self.test_energies.squeeze()), interpolated_data)
        return experiment

    def get_gp_interpolation_for_spectrum_index(self, ind):
        instrument = SimulatedMeasurement(data=(to_numpy(self.test_energies.view(1, -1)),
                                                self.test_data_all_spectra[ind]))
        data_y = instrument.measure(self.point_grid)
        data_y = torch.tensor(data_y).view(-1, 1)
        interpolated_data = self.get_gp_interpolation(self.point_grid, data_y)
        interpolated_data = to_numpy(interpolated_data.squeeze())
        if self.save_plots:
            self.save_spectrum_estimate_plot_and_data(ind, to_numpy(self.test_energies.squeeze()), interpolated_data)
        return interpolated_data

    def save_spectrum_estimate_plot_and_data(self, ind, energies, estimated_data):
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_numpy(self.test_energies.squeeze()), estimated_data, label='Estimated spectrum')
        ax.plot(to_numpy(self.test_energies.squeeze()), self.test_data_all_spectra[ind], color='gray',
                linestyle='--', label='Actual spectrum')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'estimated_data_ind_{}.pdf'.format(ind)))
        plt.close(fig)

        df = pd.DataFrame(data={
            'energy': energies,
            'estimated_data': estimated_data,
            'true_data': self.test_data_all_spectra[ind]
        })
        df.to_csv(os.path.join(self.output_dir, 'estimated_data_ind_{}.csv'.format(ind)), index=False)

    def get_gp_interpolation(self, data_x, data_y):
        configs = self.get_generic_config()
        guide = autobl.steering.guide.XANESExperimentGuide(configs)
        guide.build(data_x, data_y)
        mu, _ = guide.get_posterior_mean_and_std(self.test_energies)
        if self.debug:
            plt.plot(self.test_energies.squeeze(), to_numpy(mu.squeeze()))
            plt.scatter(data_x.squeeze(), data_y.squeeze())
            plt.show()
        return mu

    def build_point_grid(self):
        experiment = self.run_acquisition_for_spectrum_index(0)
        self.point_grid, _ = experiment.guide.untransform_data(x=experiment.guide.data_x)
        self.point_grid = torch.sort(self.point_grid, dim=0).values

    def log_data(self, ind, rms):
        self.results['spectrum_index'].append(ind)
        self.results['rms'].append(rms)

    def build(self):
        self.load_data()

    def run(self):
        self.build_point_grid()
        for ind in range(1, len(self.test_data_all_spectra)):
            data_interp = self.get_gp_interpolation_for_spectrum_index(ind)
            metric_val = rms(data_interp, self.test_data_all_spectra[ind])
            self.log_data(ind, metric_val)

    def post_analyze(self):
        plt.savefig(os.path.join(self.output_dir, 'rms_all_test_spectra.pdf'))
        self.analyze_phase_transition_percentages()

    def analyze_rms(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.results['spectrum_index'], self.results['rms'])
        ax.set_xlabel('Spectrum')
        ax.set_ylabel('RMS')

    def analyze_phase_transition_percentages(self):
        indices = []
        percentages_estimated = []
        percentages_true = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
        for f in flist:
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table = pd.read_csv(f, index_col=None)
            estimated_spectrum = table['estimated_data'].to_numpy()
            true_spectrum = table['true_data'].to_numpy()
            p_estimated = self.get_phase_transition_percentage(estimated_spectrum)
            percentages_estimated.append(p_estimated)
            p_true = self.get_phase_transition_percentage(true_spectrum)
            percentages_true.append(p_true)
        fig, ax = plt.subplots(1, 1)
        ax.plot(indices, percentages_estimated, label='Estimated')
        ax.plot(indices, percentages_true, label='True')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'phase_transition_percentages.pdf'))

    def get_phase_transition_percentage(self, data):
        amat = self.ref_spectra_y.T
        bvec = data.reshape(-1, 1)
        xvec = np.matmul(np.linalg.pinv(amat), bvec)
        w = xvec.reshape(-1)
        p = w[-1] / np.sum(w)
        return p


if __name__ == '__main__':
    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample1_50C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_initOfSelf/Sample1_50C'
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample2_60C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_initOfSelf/Sample2_60C'
    )
    tester.build()
    tester.run()
    tester.post_analyze()
