import os
import glob
import pickle

import matplotlib.pyplot as plt
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

    def __init__(self, guide_configs, name, *args, **kwargs):
        super().__init__()
        self.guide_confgis = guide_configs
        self.name = name


class SimulatedScanningExperiment(ScanningExperiment):

    def __init__(self, guide_configs, name, run_analysis=True, *args, **kwargs):
        super().__init__(guide_configs, name, *args, **kwargs)
        self.data_x = None
        self.data_y = None
        self.guide = None
        self.instrument = None
        self.analyzer = None
        self.n_pts_measured = 0
        self.run_analysis = run_analysis

    def build(self, true_data_x, true_data_y):
        self.build_instrument(true_data_x, true_data_y)

    def build_instrument(self, true_data_x, true_data_y):
        self.instrument = SimulatedMeasurement(data=(true_data_x[None, :], true_data_y))
        self.data_x = true_data_x
        self.data_y = true_data_y

    def initialize_guide(self, x_init, y_init):
        self.guide = autobl.steering.guide.XANESExperimentGuide(self.guide_confgis)
        self.guide.build(x_init, y_init)

    def initialize_analyzer(self, n_plot_interval, n_target_measurements):
        self.analyzer = ScanningExperimentAnalyzer(self.guide, self.name, self.data_x, self.data_y,
                                                   n_target_measurements, n_plot_interval)
        self.analyzer.enable(self.run_analysis)

    def take_initial_measurements(self, n):
        x_init = torch.linspace(self.data_x[0], self.data_x[-1], n).double().reshape(-1, 1)
        y_init = self.instrument.measure(x_init).reshape(-1, 1)
        self.n_pts_measured += n
        return x_init, y_init

    def run(self, n_initial_measurements=10, n_target_measurements=70, n_plot_interval=5):
        x_init, y_init = self.take_initial_measurements(n_initial_measurements)
        self.initialize_guide(x_init, y_init)
        self.initialize_analyzer(n_plot_interval, n_target_measurements)
        self.analyzer.increment_n_points_measured(n_initial_measurements)
        # self.analyzer.plot_data(additional_x=x_init, additional_y=y_init)

        if n_target_measurements is None:
            n_target_measurements = len(self.data_x) - n_initial_measurements

        for i in tqdm.trange(n_target_measurements):
            candidates = self.guide.suggest().double()
            y_new = self.instrument.measure(candidates).unsqueeze(-1)
            self.guide.update(candidates, y_new)
            self.n_pts_measured += len(candidates)
            self.analyzer.increment_n_points_measured(by=len(candidates))
            self.analyzer.update_analysis()

        self.analyzer.save_analysis()
