import os
import glob
import pickle
import logging

import matplotlib.pyplot as plt
import scipy.interpolate
import torch
import numpy as np
import pandas as pd
import tqdm

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.steering.acquisition import *
from autobl.steering.optimization import *
from autobl.steering.analysis import ScanningExperimentAnalyzer
from autobl.util import *


class Experiment:

     def __init__(self, *args, **kwargs):
         pass


class ScanningExperiment(Experiment):

    def __init__(self, guide_configs, *args, **kwargs):
        super().__init__()
        self.guide_configs = guide_configs
        self.candidate_list = []


class SimulatedScanningExperiment(ScanningExperiment):

    def __init__(self, guide_configs, guide_class=autobl.steering.guide.XANESExperimentGuide,
                 run_analysis=True, analyzer_configs=None,
                 auto_narrow_down_scan_range=False, narrow_down_range_bounds_ev=(-70, 90),
                 *args, **kwargs):
        super().__init__(guide_configs, *args, **kwargs)
        self.data_x = None
        self.data_y = None
        self.data_x_truncated = None
        self.data_y_truncated = None
        self.data_x_measured = torch.tensor([])
        self.data_y_measured = torch.tensor([])
        self.guide = None
        self.instrument = None
        self.analyzer = None
        self.analyzer_configs = analyzer_configs
        self.n_pts_measured = 0
        self.run_analysis = run_analysis
        self.select_scan_range = auto_narrow_down_scan_range
        self.range_bounds_ev = narrow_down_range_bounds_ev
        self.guide_class = guide_class

    def build(self, true_data_x, true_data_y):
        self.build_instrument(true_data_x, true_data_y)

    def build_instrument(self, true_data_x, true_data_y):
        self.instrument = SimulatedMeasurement(data=(true_data_x[None, :], true_data_y))
        self.data_x = true_data_x
        self.data_y = true_data_y
        self.data_x_truncated = self.data_x
        self.data_y_truncated = self.data_y

    def initialize_guide(self, x_init, y_init):
        self.guide = self.guide_class(self.guide_configs)
        self.guide.build(x_init, y_init)

    def initialize_analyzer(self, analyzer_configs, n_target_measurements, n_initial_measurements):
        if analyzer_configs is None:
            analyzer_configs = ExperimentAnalyzerConfig()
        self.analyzer = ScanningExperimentAnalyzer(analyzer_configs, self.guide,
                                                   self.data_x_truncated, self.data_y_truncated,
                                                   n_target_measurements=n_target_measurements,
                                                   n_init_measurements=n_initial_measurements)
        self.analyzer.enable(self.run_analysis)

    def record_data(self, data_x, data_y):
        self.data_x_measured = torch.concatenate([self.data_x_measured, data_x])
        self.data_y_measured = torch.concatenate([self.data_y_measured, data_y])

    def get_initial_measurement_locations(self, n, method='uniform'):
        lb, ub = self.guide_configs.lower_bounds[0], self.guide_configs.upper_bounds[0]
        if method == 'uniform':
            x_init = torch.linspace(lb, ub, n).double().reshape(-1, 1)
        elif method == 'random':
            x_init = torch.rand(n) * (ub - lb) + lb
            x_init = torch.sort(x_init).values
            x_init = x_init.double().reshape(-1, 1)
        elif method == 'spectral':
            spacing_low = 2
            spacing_high = 14
            logging.info('Set value of number of initial measurements is disregarded because "spectral" is '
                         'chosen as the method to generate initial measurement locations.')
            x_init = torch.tensor([lb])
            spacings = torch.arange(spacing_low, spacing_high + 0.1, 1.0)
            while x_init[-1] <= ub:
                x_init = torch.concatenate([x_init, x_init[-1] + torch.cumsum(spacings, dim=0)])
            x_init = x_init[x_init <= ub]
            x_init = x_init.double().reshape(-1, 1)
            logging.info('Generated {} initial measurement locations.'.format(len(x_init)))
        else:
            raise ValueError('{} is not a valid method to generate initial locations.'.format(method))
        return x_init

    def take_initial_measurements(self, n, method='random'):
        x_init = self.get_initial_measurement_locations(n, method=method)
        y_init = self.instrument.measure(x_init).reshape(-1, 1)
        self.n_pts_measured += n
        self.record_data(x_init, y_init)
        return x_init, y_init

    def update_candidate_list(self, candidates):
        self.candidate_list.append(candidates.squeeze().detach().cpu().numpy())

    def adjust_scan_range_and_init_data(self, x_init, y_init):
        temp_guide = autobl.steering.guide.XANESExperimentGuide(self.guide_configs)
        edge_loc, edge_width = temp_guide.estimate_edge_location_and_width(
            x_init, y_init, input_is_transformed=False, run_in_transformed_space=False, return_normalized_values=False)
        new_range = (edge_loc + self.range_bounds_ev[0], edge_loc + self.range_bounds_ev[1])
        logging.info('Edge detected at {} eV.'.format(edge_loc))
        logging.info('Scan range adjusted to {}.'.format(new_range))
        logging.info('Config object is being modified.'.format(new_range))
        self.guide_configs.lower_bounds = torch.tensor([new_range[0]])
        self.guide_configs.upper_bounds = torch.tensor([new_range[1]])
        logging.info('Removing initial measurements outside the new range...')
        inds_keep = torch.where((x_init > new_range[0]) & (x_init < new_range[1]))[0]
        x_init = x_init[inds_keep]
        y_init = y_init[inds_keep]
        if 'reference_spectra_x' in self.guide_configs.acquisition_function_params.keys():
            orig_x = self.guide_configs.acquisition_function_params['reference_spectra_x']
            inds_keep = torch.where((orig_x > new_range[0]) & (orig_x < new_range[1]))[0]
            self.guide_configs.acquisition_function_params['reference_spectra_x'] = (
                self.guide_configs.acquisition_function_params)['reference_spectra_x'][inds_keep]
            self.guide_configs.acquisition_function_params['reference_spectra_y'] = (
                self.guide_configs.acquisition_function_params)['reference_spectra_y'][:, inds_keep]

        logging.info('Narrowing down reference spectra...')
        inds_keep = torch.where((x_init > new_range[0]) & (x_init < new_range[1]))[0]
        x_init = x_init[inds_keep]
        y_init = y_init[inds_keep]

        logging.info('Narrowing down true data...')
        inds_keep = torch.where((self.data_x > new_range[0]) & (self.data_x < new_range[1]))[0]
        self.data_x_truncated = self.data_x[inds_keep]
        self.data_y_truncated = self.data_y[inds_keep]
        return x_init, y_init

    def get_measured_data(self, sort=True):
        x = self.data_x_measured
        y = self.data_y_measured
        if sort:
            sorted_inds = torch.argsort(x, dim=0)
            x = x[sorted_inds]
            y = y[sorted_inds]
        return x, y

    def get_estimated_spectrum(self, x):
        """
        Get estimated spectrum based on data measured.

        Input x may contain data outside the range that is actually guided by adaptive sampler. In that case,
        queried points outside the range are spline interpolated from measured data. For queried points within the
        guided range, the posterior mean is used instead.

        :param x: np.ndarray | Tensor. Points where the estimated spectrum values are to be queried.
        :return: np.ndarray. Estimated spectrum.
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        x = to_tensor(x)

        # First, run cubic spline interpolation for all data
        x_dat = to_numpy(x.squeeze())
        x_measured, y_measured = self.get_measured_data(sort=True)
        y_dat = scipy.interpolate.CubicSpline(to_numpy(x_measured.squeeze()), to_numpy(y_measured.squeeze()))(x_dat)

        # Second, get posterior mean (which may also be spline interpolated depending on setting) for points
        # within the range that is guided by adaptive sampler.
        inds_in_range = torch.where((x > self.guide.config.lower_bounds[0]) & (x < self.guide.config.upper_bounds[0]))[0]
        x_in_range = x[inds_in_range]
        y_in_range, _ = self.guide.get_posterior_mean_and_std(x_in_range, transform=True, untransform=True)
        y_in_range = to_numpy(y_in_range.squeeze())
        y_dat[inds_in_range] = y_in_range
        return y_dat

    def run(self, n_initial_measurements=10, n_target_measurements=70, initial_measurement_method='uniform'):
        x_init, y_init = self.take_initial_measurements(n_initial_measurements, method=initial_measurement_method)
        if self.select_scan_range:
            x_init, y_init = self.adjust_scan_range_and_init_data(x_init, y_init)
        self.initialize_guide(x_init, y_init)
        self.initialize_analyzer(self.analyzer_configs, n_target_measurements, n_initial_measurements)
        self.analyzer.increment_n_points_measured(n_initial_measurements)
        self.analyzer.update_analysis()
        # self.analyzer.plot_data(additional_x=x_init, additional_y=y_init)

        if n_target_measurements is None:
            n_target_measurements = len(self.data_x)
        for i in tqdm.trange(n_initial_measurements, n_target_measurements):
            candidates = self.guide.suggest().double()
            self.update_candidate_list(candidates)
            y_new = self.instrument.measure(candidates).unsqueeze(-1)
            self.guide.update(candidates, y_new)
            self.n_pts_measured += len(candidates)
            self.analyzer.increment_n_points_measured(by=len(candidates))
            self.analyzer.update_analysis()
            if self.guide.stopping_criterion.check():
                break

        self.record_data(*self.guide.untransform_data(self.guide.data_x[len(x_init):], self.guide.data_y[len(y_init):]))
        self.analyzer.save_analysis()
