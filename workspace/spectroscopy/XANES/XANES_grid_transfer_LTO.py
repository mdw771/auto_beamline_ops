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
from autobl.util import *

torch.set_default_device('cpu')

set_random_seed(123)


class LTOGridTransferTester:

    def __init__(self, test_data_path, ref_spectra_data_path, output_dir='outputs',
                 grid_generation_method='init', grid_generation_spectra_indices=(0,), grid_intersect_tol=1.0,
                 n_initial_measurements=10, n_target_measurements=40, initialization_method='uniform'):
        self.test_data_path = test_data_path
        self.ref_spectra_data_path = ref_spectra_data_path
        self.test_data_all_spectra = None
        self.test_energies = None
        self.ref_data_all_spectra = None
        self.ref_spectra_x = None
        self.ref_spectra_y = None
        self.output_dir = output_dir
        self.results = {'spectrum_index': [], 'rms': []}
        self.point_grid = None
        self.grid_generation_method = grid_generation_method
        self.grid_generation_spectra_indices = grid_generation_spectra_indices
        self.grid_intersect_tol = grid_intersect_tol
        self.n_initial_measurements = n_initial_measurements
        self.n_target_measurements = n_target_measurements
        self.initialization_method = initialization_method
        self.save_plots = True
        self.debug = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_metadata(self):
        d = {
            'test_data_path': self.test_data_path,
            'ref_spectra_data_path': self.ref_spectra_data_path,
            'output_dir': self.output_dir,
            'grid_generation_method': self.grid_generation_method,
            'grid_generation_spectra_indices': self.grid_generation_spectra_indices,
            'grid_intersect_tol': self.grid_intersect_tol,
            'n_initial_measurements': self.n_initial_measurements,
            'n_target_measurements': self.n_target_measurements
        }
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(d, f, indent=4, separators=(',', ': '))

    @staticmethod
    def load_data_from_csv(path):
        table = pd.read_csv(path, header=None)
        data_all_spectra = table.iloc[1:].to_numpy()
        energies = torch.tensor(table.iloc[0].to_numpy())
        return data_all_spectra, energies

    def load_data(self):
        self.test_data_all_spectra, self.test_energies = self.load_data_from_csv(self.test_data_path)

        self.ref_data_all_spectra, self.ref_spectra_x = self.load_data_from_csv(self.ref_spectra_data_path)
        ref_spectra_0 = torch.tensor(self.ref_data_all_spectra[0])
        ref_spectra_1 = torch.tensor(self.ref_data_all_spectra[-1])
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
            stopping_criterion_configs=StoppingCriterionConfig(
                method='max_uncertainty',
                params={'threshold': 0.02},
                n_updates_to_begin=6,
            ),
            use_spline_interpolation_for_posterior_mean=True
        )
        return configs

    def run_acquisition_for_spectrum_index(self, ind, name_prefix='LTO_50C', dataset='test'):
        configs = self.get_generic_config()

        analyzer_configs = ExperimentAnalyzerConfig(
            name='{}_index_{}'.format(name_prefix, ind),
            output_dir=self.output_dir,
            n_plot_interval=5,
            save=self.save_plots
        )

        if dataset == 'test':
            data = self.test_data_all_spectra[ind]
            energies = self.test_energies
        elif dataset == 'ref':
            data = self.ref_data_all_spectra[ind]
            energies = self.ref_spectra_x
        else:
            raise ValueError
        experiment = SimulatedScanningExperiment(configs, run_analysis=True, analyzer_configs=analyzer_configs)
        experiment.build(self.test_energies, data)
        experiment.run(n_initial_measurements=self.n_initial_measurements,
                       n_target_measurements=self.n_target_measurements,
                       initial_measurement_method=self.initialization_method)
        if self.save_plots and dataset == 'test':
            x_measured, y_measured = experiment.guide.untransform_data(x=experiment.guide.data_x,
                                                                       y=experiment.guide.data_y)
            x_measured, y_measured = to_numpy(x_measured.squeeze()), to_numpy(y_measured.squeeze())
            interpolated_data = to_numpy(experiment.guide.get_posterior_mean_and_std(energies)[0].squeeze())
            self.save_spectrum_estimate_plot_and_data(ind, to_numpy(energies.squeeze()), interpolated_data,
                                                      x_measured=x_measured,
                                                      y_measured=y_measured)
        return experiment

    def get_gp_interpolation_for_spectrum_index(self, ind):
        instrument = SimulatedMeasurement(data=(to_numpy(self.test_energies.view(1, -1)),
                                                self.test_data_all_spectra[ind]))
        data_y = instrument.measure(self.point_grid)
        data_y = torch.tensor(data_y).view(-1, 1)
        interpolated_data = self.get_gp_interpolation(self.point_grid, data_y)
        interpolated_data = to_numpy(interpolated_data.squeeze())
        if self.save_plots:
            self.save_spectrum_estimate_plot_and_data(ind, to_numpy(self.test_energies.squeeze()), interpolated_data,
                                                      x_measured=to_numpy(self.point_grid.squeeze()),
                                                      y_measured=to_numpy(data_y.squeeze()))
        return interpolated_data

    def save_spectrum_estimate_plot_and_data(self, ind, energies, estimated_data, x_measured=None, y_measured=None):
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_numpy(self.test_energies.squeeze()), estimated_data, label='Estimated spectrum')
        ax.plot(to_numpy(self.test_energies.squeeze()), self.test_data_all_spectra[ind], color='gray',
                linestyle='--', label='Actual spectrum')
        if x_measured is not None and y_measured is not None:
            ax.scatter(x_measured, y_measured, s=3)
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'estimated_data_ind_{}.pdf'.format(ind)))
        plt.close(fig)

        df = pd.DataFrame(data={
            'energy': energies,
            'estimated_data': estimated_data,
            'true_data': self.test_data_all_spectra[ind],
        })
        df.to_csv(os.path.join(self.output_dir, 'estimated_data_ind_{}.csv'.format(ind)), index=False)

        if x_measured is not None and y_measured is not None:
            df = pd.DataFrame(data={
                'x_measured': x_measured,
                'y_measured': y_measured,
            })
            df.to_csv(os.path.join(self.output_dir, 'measured_data_ind_{}.csv'.format(ind)), index=False)

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

    def build_point_grid(self, spectrum_ind=None):
        if self.grid_generation_method == 'init':
            experiment = self.run_acquisition_for_spectrum_index(0)
            self.point_grid, _ = experiment.guide.untransform_data(x=experiment.guide.data_x)
            self.point_grid = torch.sort(self.point_grid, dim=0).values
        elif self.grid_generation_method == 'ref':
            ref_points = []
            for ref_ind in self.grid_generation_spectra_indices:
                experiment = self.run_acquisition_for_spectrum_index(ref_ind, dataset='ref')
                point_grid = to_numpy(experiment.guide.untransform_data(x=experiment.guide.data_x)[0].squeeze())
                ref_points.append(point_grid)
            intersect_point_grid = self.find_intersects_with_tolerance_multi_arrays(ref_points,
                                                                                    tol=self.grid_intersect_tol)
            logging.info('Numbers of points from all reference spectra: {}'.format([len(x) for x in ref_points]))
            logging.info('The merged grid has {} points.'.format(len(intersect_point_grid)))
            self.point_grid = torch.tensor(intersect_point_grid.reshape(-1, 1))
            self.point_grid = torch.sort(self.point_grid, dim=0).values
        elif self.grid_generation_method == 'redo_for_each':
            assert spectrum_ind is not None
            experiment = self.run_acquisition_for_spectrum_index(spectrum_ind)
            self.point_grid, _ = experiment.guide.untransform_data(x=experiment.guide.data_x)
            self.point_grid = torch.sort(self.point_grid, dim=0).values
        else:
            raise ValueError('{} is invalid.'.format(self.grid_generation_method))

    @staticmethod
    def find_intersects_with_tolerance(arr1, arr2, tol=1.0):
        dist = np.abs(arr1.reshape(-1, 1) - arr2).min(1)
        ind_in_1 = np.where(dist < tol)
        return arr1[ind_in_1]

    def find_intersects_with_tolerance_multi_arrays(self, arrs, tol=1.0):
        intersect = arrs[0]
        for arr in arrs[1:]:
            intersect = self.find_intersects_with_tolerance(intersect, arr, tol=tol)
        return intersect

    def log_data(self, ind, rms):
        self.results['spectrum_index'].append(ind)
        self.results['rms'].append(rms)

    def build(self):
        self.load_data()
        self.save_metadata()

    def run(self):
        if self.grid_generation_method != 'redo_for_each':
            self.build_point_grid()
        for ind in range(len(self.test_data_all_spectra)):
            if self.grid_generation_method == 'redo_for_each':
                self.build_point_grid(spectrum_ind=ind)
            data_interp = self.get_gp_interpolation_for_spectrum_index(ind)
            metric_val = rms(data_interp, self.test_data_all_spectra[ind])
            self.log_data(ind, metric_val)

    def post_analyze(self):
        self.analyze_rms()
        self.analyze_phase_transition_percentages()

    def analyze_rms(self):
        indices = []
        rms_list = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]
        for f in flist:
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table = pd.read_csv(f, index_col=None)
            estimated_spectrum = table['estimated_data'].to_numpy()
            true_spectrum = table['true_data'].to_numpy()
            metric_val = rms(estimated_spectrum, true_spectrum)
            rms_list.append(metric_val)

        np.savetxt(os.path.join(self.output_dir, 'rms_all_test_spectra.txt'), rms_list)

        fig, ax = plt.subplots(1, 1)
        ax.plot(indices, rms_list)
        ax.set_xlabel('Spectrum')
        ax.set_ylabel('RMS')
        plt.savefig(os.path.join(self.output_dir, 'rms_all_test_spectra.pdf'))

    def calculate_phase_transition_percentages(self):
        indices = []
        percentages_estimated = []
        percentages_true = []
        fitting_residue_estimated = []
        flist = glob.glob(os.path.join(self.output_dir, 'estimated_data_ind_*.csv'))
        flist = np.array(flist)[np.argsort([int(re.findall('\d+', x)[-1]) for x in flist])]

        ref_spectrum_fitting_estimated = np.stack([
            pd.read_csv(flist[0], index_col=None)['estimated_data'].to_numpy(),
            pd.read_csv(flist[-1], index_col=None)['estimated_data'].to_numpy(),
        ])
        ref_spectrum_fitting_true = np.stack([
            pd.read_csv(flist[0], index_col=None)['true_data'].to_numpy(),
            pd.read_csv(flist[-1], index_col=None)['true_data'].to_numpy(),
        ])

        for f in flist:
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table = pd.read_csv(f, index_col=None)
            estimated_spectrum = table['estimated_data'].to_numpy()
            true_spectrum = table['true_data'].to_numpy()
            p_estimated, r = self.get_phase_transition_percentage(estimated_spectrum, ref_spectrum_fitting_estimated,
                                                                  return_fitting_residue=True)
            percentages_estimated.append(p_estimated)
            fitting_residue_estimated.append(r)
            p_true = self.get_phase_transition_percentage(true_spectrum, ref_spectrum_fitting_true)
            percentages_true.append(p_true)

        table = pd.DataFrame(data={'indices': indices,
                                   'percentages_estimated': percentages_estimated,
                                   'percentages_true': percentages_true,
                                   'fitting_residue_estimated': fitting_residue_estimated
                                   })
        return table

    def analyze_phase_transition_percentages(self):
        table = self.calculate_phase_transition_percentages()
        table.to_csv(os.path.join(self.output_dir, 'phase_transition_percentages.csv'), index=False)

        fig, ax = plt.subplots(1, 1)
        ax.plot(table['indices'], table['percentages_estimated'], label='Estimated')
        ax.plot(table['indices'], table['percentages_true'], label='True')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'phase_transition_percentages.pdf'))

        fig, ax = plt.subplots(1, 1)
        ax.plot(table['indices'], table['fitting_residue_estimated'])
        fig.savefig(os.path.join(self.output_dir, 'fitting_residue.pdf'))

    @staticmethod
    def get_phase_transition_percentage(data, reference_spectra_for_fitting, return_fitting_residue=False, add_constant_term=False):
        amat = to_numpy(reference_spectra_for_fitting).T
        if add_constant_term:
            amat = np.concatenate(amat, np.ones([amat.shape[0], 1]))
        bvec = data.reshape(-1, 1)
        xvec = np.matmul(np.linalg.pinv(amat), bvec)
        w = xvec.reshape(-1)
        p = w[-1] / np.sum(w)
        if return_fitting_residue:
            y_fit = (amat @ xvec).reshape(-1)
            residue = np.mean((y_fit - data) ** 2)
            return p, residue
        return p


if __name__ == '__main__':
    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample1_50C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_redoForEach/Sample1_50C',
        grid_generation_method='redo_for_each',
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample2_60C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_redoForEach/Sample2_60C',
        grid_generation_method='redo_for_each',
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample1_50C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_initOfSelf/Sample1_50C',
        grid_generation_method='init',
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample2_60C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_initOfSelf/Sample2_60C',
        grid_generation_method='init',
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample1_50C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_selectedRef/Sample1_50C',
        grid_generation_method='ref',
        grid_generation_spectra_indices=(0, 5, 8, 12),
        grid_intersect_tol=3.0,
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample2_60C_XANES.csv',
        ref_spectra_data_path='data/raw/LiTiO_XANES/dataanalysis/Originplots/Sample3_70C_XANES.csv',
        output_dir='outputs/grid_transfer/grid_selectedRef/Sample2_60C',
        grid_generation_method='ref',
        grid_generation_spectra_indices=(0, 5, 8, 12),
        grid_intersect_tol=3.0,
        n_initial_measurements=10, n_target_measurements=40
    )
    tester.build()
    tester.run()
    tester.post_analyze()
