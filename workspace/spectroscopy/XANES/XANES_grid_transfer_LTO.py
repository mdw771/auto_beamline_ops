import os
import glob
import pickle
import re
import logging
import json

logging.basicConfig(level=logging.INFO)

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
from autobl.util import to_numpy, to_tensor, generate_quasi_random_numbers
import autobl.tools.spectroscopy.xanes as xanestools
from autobl.steering.util import estimate_noise_variance

from plot_results import interpolate_data_on_grid

torch.set_default_device('cpu')


class LTOGridTransferTester:

    def __init__(self, test_data_path, test_data_filename_pattern, ref_spectra_data_path, ref_data_filename_pattern, output_dir='outputs',
                 grid_generation_method='init', grid_generation_spectra_indices=(0,), grid_intersect_tol=1.0,
                 n_initial_measurements=10, n_target_measurements=40, initialization_method='uniform'):
        self.test_data_path = test_data_path
        self.ref_spectra_data_path = ref_spectra_data_path
        self.test_data_filename_pattern = test_data_filename_pattern
        self.ref_data_filename_pattern = ref_data_filename_pattern
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
        self.supplied_initial_points = None
        self.save_plots = True
        self.debug = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_metadata(self):
        d = {
            'test_data_path': self.test_data_path,
            'ref_spectra_data_path': self.ref_spectra_data_path,
            'test_data_filename_pattern': self.test_data_filename_pattern,
            'ref_data_filename_pattern': self.ref_data_filename_pattern,
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
    def load_data_from_csv(path, filename_pattern, normalizer=False, crop=True):
        dataset = LTORawDataset(path, filename_pattern=filename_pattern)
        energies = dataset.energies_ev
        data_all_spectra = dataset.data
        if normalizer is not None:
            for i in range(len(data_all_spectra)):
                normalizer.fit(energies, data_all_spectra[i])
                data_all_spectra[i] = normalizer.apply(energies, data_all_spectra[i])
        if crop:
            mask = (energies >= 4936) & (energies <= 5006)
            data_all_spectra = data_all_spectra[:, mask]
            energies = energies[mask]
        energies = torch.tensor(energies)
        return data_all_spectra, energies

    def load_data(self, normalizer=None):
        self.test_data_all_spectra_raw, self.test_energies_raw = self.load_data_from_csv(self.test_data_path, filename_pattern=self.test_data_filename_pattern, normalizer=None, crop=False)
        self.ref_data_all_spectra_raw, self.ref_spectra_x_raw = self.load_data_from_csv(self.ref_spectra_data_path, filename_pattern=self.ref_data_filename_pattern, normalizer=None, crop=False)
        self.test_data_all_spectra, self.test_energies = self.load_data_from_csv(self.test_data_path, filename_pattern=self.test_data_filename_pattern, normalizer=normalizer)
        self.ref_data_all_spectra, self.ref_spectra_x = self.load_data_from_csv(self.ref_spectra_data_path, filename_pattern=self.ref_data_filename_pattern, normalizer=normalizer)
        ref_spectra_0 = torch.tensor(self.ref_data_all_spectra[0])
        ref_spectra_1 = torch.tensor(self.ref_data_all_spectra[-1])
        self.ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)

        if self.save_plots:
            self._plot_data()

    def create_initial_points(self):
        energies = to_numpy(self.test_energies).reshape(-1)
        # self.supplied_initial_points = np.random.rand(self.n_initial_measurements) * (energies[-1] - energies[0]) + energies[0]
        # self.supplied_initial_points = np.sort(self.supplied_initial_points)
        # self.supplied_initial_points = to_tensor(generate_quasi_random_numbers(self.n_initial_measurements, energies[0], energies[-1])).double().reshape(-1, 1)
        

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
        noise_variance = estimate_noise_variance(to_numpy(self.ref_spectra_x[self.ref_spectra_x < 4960]), to_numpy(self.ref_spectra_y[0][self.ref_spectra_x < 4960]))
        print(f'Noise variance: {noise_variance}')
        
        configs = XANESExperimentGuideConfig(
            dim_measurement_space=1,
            num_candidates=1,
            model_class=botorch.models.SingleTaskGP,
            model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
            # reference_spectra_for_lengthscale_fitting=(self.ref_spectra_x, self.ref_spectra_y[0]),
            override_kernel_lengthscale=9.1,
            noise_variance=noise_variance,
            adaptive_noise_variance=True,
            lower_bounds=torch.tensor([self.test_energies[0]]),
            upper_bounds=torch.tensor([self.test_energies[-1]]),
            acquisition_function_class=ComprehensiveAugmentedAcquisitionFunction,
            acquisition_function_params={'gradient_order': 2,
                                         'differentiation_method': 'numerical',
                                         'reference_spectra_x': self.ref_spectra_x,
                                         'reference_spectra_y': self.ref_spectra_y,
                                         'phi_r': 1e2,
                                         'phi_g': 1e-1,
                                         'phi_g2': 1e-3,
                                         'beta': 0.999,
                                         'gamma': 0.95,
                                         'addon_term_lower_bound': 3e-2,
                                         'estimate_posterior_mean_by_interpolation': True,
                                         'debug': False},
            n_updates_create_acqf_weight_func=5,
            acqf_weight_func_floor_value=0.01,
            acqf_weight_func_post_edge_gain=3.0,
            # acqf_weight_func_edge_offset=-2.5,
            optimizer_class=DiscreteOptimizer,
            optimizer_params={'optim_func': botorch.optim.optimize.optimize_acqf_discrete,
                              'optim_func_params': {
                                  'choices': torch.linspace(0, 1, 1000)[:, None]
                              }
                              },
            stopping_criterion_configs=StoppingCriterionConfig(
                method='max_uncertainty',
                n_max_measurements=40,
                params={'threshold': 0.001},
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
                       initial_measurement_method=self.initialization_method,
                       supplied_initial_points=self.supplied_initial_points)
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
        if len(self.test_data_all_spectra[ind]) == len(energies):
            actual_data = self.test_data_all_spectra[ind]
        else:
            actual_data = estimated_data
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_numpy(energies.squeeze()), estimated_data, label='Estimated spectrum')
        ax.plot(to_numpy(energies.squeeze()), actual_data, color='gray',
                linestyle='--', label='Actual spectrum')
        if x_measured is not None and y_measured is not None:
            ax.scatter(x_measured, y_measured, s=3)
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'estimated_data_ind_{}.pdf'.format(ind)))
        plt.close(fig)
        
        df = pd.DataFrame(data={
            'energy': energies,
            'estimated_data': estimated_data,
            'true_data': actual_data,
        })
        df.to_csv(os.path.join(self.output_dir, 'estimated_data_ind_{}.csv'.format(ind)), index=False)

        if x_measured is not None and y_measured is not None:
            df = pd.DataFrame(data={
                'x_measured': x_measured,
                'y_measured': y_measured,
            })
            df.to_csv(os.path.join(self.output_dir, 'measured_data_ind_{}.csv'.format(ind)), index=False)

    def get_gp_interpolation(self, data_x, data_y, query_x=None):
        if query_x is None:
            query_x = self.test_energies
        configs = self.get_generic_config()
        guide = autobl.steering.guide.XANESExperimentGuide(configs)
        guide.build(data_x, data_y)
        mu, _ = guide.get_posterior_mean_and_std(query_x)
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
            intersect_point_grid = find_intersects_with_tolerance_multi_arrays(ref_points,
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
        self.point_grid = torch.unique(self.point_grid, dim=0, sorted=True)

    def log_data(self, ind, rms):
        self.results['spectrum_index'].append(ind)
        self.results['rms'].append(rms)

    def build(self, normalizer=None):
        self.load_data(normalizer=normalizer)
        if self.initialization_method == 'supplied':
            self.create_initial_points()
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

    def post_analyze(self, normalizer=None, normalize_percentages=False):
        self.analyze_rms()
        self.analyze_phase_transition_percentages(normalizer=normalizer, normalize_percentages=normalize_percentages)

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
            energies = table['energy'].to_numpy()
            
            try:
                # Interpolate on dense grid
                dict_file = glob.glob(os.path.join(self.output_dir, "*index_{}*.pkl".format(ind)))[0]
                with open(dict_file, 'rb') as f:
                    d = pickle.load(f)
                dense_energy = d["x_dense_list"]
                dense_estimated_spectrum = d["mu_dense_list"][-1]
                _, dense_true_spectrum = interpolate_data_on_grid(energies, true_spectrum, len(dense_energy))
                
                metric_val = rms(dense_estimated_spectrum, dense_true_spectrum)
            except:
                logging.warning('No dense grid data found for spectrum index {}.'.format(ind))
                metric_val = rms(estimated_spectrum, true_spectrum)
            rms_list.append(metric_val)

        np.savetxt(os.path.join(self.output_dir, 'rms_all_test_spectra.txt'), rms_list)

        fig, ax = plt.subplots(1, 1)
        ax.plot(indices, rms_list)
        ax.set_xlabel('Spectrum')
        ax.set_ylabel('RMS')
        plt.savefig(os.path.join(self.output_dir, 'rms_all_test_spectra.pdf'))

    def calculate_phase_transition_percentages(self, normalizer=None, normalize_percentages=False):
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
        
        # Normalize and detilt if normalizer is provided
        if normalizer is not None:
            normalizer.fit(to_numpy(self.test_energies_raw), self.test_data_all_spectra_raw[0])
            ref_spectrum_fitting_estimated[0] = normalizer.apply(self.ref_spectra_x, ref_spectrum_fitting_estimated[0])
            ref_spectrum_fitting_true[0] = normalizer.apply(self.ref_spectra_x, ref_spectrum_fitting_true[0])
            normalizer.fit(to_numpy(self.test_energies_raw), self.test_data_all_spectra_raw[-1])
            ref_spectrum_fitting_estimated[1] = normalizer.apply(self.ref_spectra_x, ref_spectrum_fitting_estimated[1])
            ref_spectrum_fitting_true[1] = normalizer.apply(self.ref_spectra_x, ref_spectrum_fitting_true[1])

        for i, f in enumerate(flist):
            ind = int(re.findall('\d+', f)[-1])
            indices.append(ind)
            table = pd.read_csv(f, index_col=None)
            energies = table['energy'].to_numpy()
            estimated_spectrum = table['estimated_data'].to_numpy()
            true_spectrum = table['true_data'].to_numpy()
            
            # Normalize and detilt if normalizer is provided
            if normalizer is not None:
                normalizer.fit(to_numpy(self.test_energies_raw), self.test_data_all_spectra_raw[i])
                estimated_spectrum = normalizer.apply(energies, estimated_spectrum)
                true_spectrum = normalizer.apply(energies, true_spectrum)
            
            p_estimated, r = self.get_phase_transition_percentage(estimated_spectrum, ref_spectrum_fitting_estimated,
                                                                  return_fitting_residue=True)
            percentages_estimated.append(p_estimated)
            fitting_residue_estimated.append(r)
            p_true = self.get_phase_transition_percentage(true_spectrum, ref_spectrum_fitting_true)
            percentages_true.append(p_true)
            
        if normalize_percentages:
            percentages_estimated = percentages_estimated / percentages_estimated.max()
            percentages_true = percentages_true / percentages_true.max()

        table = pd.DataFrame(data={'indices': indices,
                                   'percentages_estimated': percentages_estimated,
                                   'percentages_true': percentages_true,
                                   'fitting_residue_estimated': fitting_residue_estimated
                                   })
        return table

    def analyze_phase_transition_percentages(self, normalizer=None, normalize_percentages=False):
        table = self.calculate_phase_transition_percentages(normalizer=normalizer, normalize_percentages=normalize_percentages)
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
        """
        Calculate the phase transition percentage of a given spectrum using
        least-squares fitting.

        :param data: A 1D array of floats representing the spectrum to be analyzed.
        :param reference_spectra_for_fitting: A 2D array of floats representing
            the spectra to be used for fitting.
        :param return_fitting_residue: A boolean indicating whether to return the
            fitting residue, defaults to False.
        :param add_constant_term: A boolean indicating whether to add a constant
            term to the fitting matrix, defaults to False.
        :return: If return_fitting_residue is False, returns a float representing
            the phase transition percentage. If return_fitting_residue is True,
            returns a tuple containing the phase transition percentage and the
            fitting residue.
        """
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
    normalizer = xanestools.XANESNormalizer(fit_ranges=((4900, 4950), (5100, 5200)), edge_loc=4983)
    
    set_random_seed(1634)
    
    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/rawdata', test_data_filename_pattern="LTOsample3.[0-9]*",
        ref_spectra_data_path='data/raw/LiTiO_XANES/rawdata', ref_data_filename_pattern="LTOsample2.[0-9]*",
        output_dir='outputs/grid_transfer_LTO/grid_redoForEach/50C',
        grid_generation_method='redo_for_each',
        n_initial_measurements=10, n_target_measurements=40, initialization_method="uniform"
    )
    tester.build()
    tester.run()
    tester.post_analyze(normalizer=normalizer)
    
    #----------------------------------------------
    
    if False:
    
        set_random_seed(1634)

        tester = LTOGridTransferTester(
            test_data_path='data/raw/LiTiO_XANES/rawdata', test_data_filename_pattern="LTOsample3.[0-9]*",
            ref_spectra_data_path='data/raw/LiTiO_XANES/rawdata', ref_data_filename_pattern="LTOsample2.[0-9]*",
            output_dir='outputs/grid_transfer_LTO/grid_initOfSelf/50C',
            grid_generation_method='init',
            n_initial_measurements=10, n_target_measurements=40, initialization_method="supplied"
        )
        tester.build()
        tester.run()
        tester.post_analyze(normalizer=normalizer)
        
        #----------------------------------------------
        
        set_random_seed(1634)

        tester = LTOGridTransferTester(
            test_data_path='data/raw/LiTiO_XANES/rawdata', test_data_filename_pattern="LTOsample3.[0-9]*",
            ref_spectra_data_path='data/raw/LiTiO_XANES/rawdata', ref_data_filename_pattern="LTOsample2.[0-9]*",
            output_dir='outputs/grid_transfer_LTO/grid_selectedRef/50C',
            grid_generation_method='ref',
            grid_generation_spectra_indices=(0, 5, 8, 12),
            grid_intersect_tol=3.0,
            n_initial_measurements=10, n_target_measurements=40, initialization_method="supplied"
        )
        tester.build()
        tester.run()
        tester.post_analyze(normalizer=normalizer)

    #------------------------------------------------
    # Generate ref data plot
    set_random_seed(1634)

    tester = LTOGridTransferTester(
        test_data_path='data/raw/LiTiO_XANES/rawdata', test_data_filename_pattern="LTOsample2.[0-9]*",
        ref_spectra_data_path='data/raw/LiTiO_XANES/rawdata', ref_data_filename_pattern="LTOsample2.[0-9]*",
        output_dir='outputs/grid_transfer_LTO/grid_initOfSelf/70C',
        grid_generation_method='init',
        n_initial_measurements=10, n_target_measurements=40, initialization_method="uniform"
    )
    tester.build()
    tester.run()
    # tester.post_analyze(normalizer=normalizer)
    