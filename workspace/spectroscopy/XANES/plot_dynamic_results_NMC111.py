import re
import glob
import os

from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from XANES_grid_transfer_NMC111 import NMC111GridTransferTester
import pandas as pd

from plot_dynamic_results import DynamicExperimentResultAnalyzer

import logging
logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)


if __name__ == '__main__':
    folders = [
        'outputs/grid_transfer_NMC111/grid_redoForEach/NMC111',
    ]
    labels = [
        'Adaptively sampled',
    ]

    normalizer = xanestools.XANESNormalizer(fit_ranges=((8234, 8312), (8385, 8481)), edge_loc=8343, normalization_order=1)
    
    analyzer = DynamicExperimentResultAnalyzer(tester_class=NMC111GridTransferTester)
    analyzer.compare_calculated_percentages(folders, labels,
                                            read_precalculated_percentage_data=True, normalizer=normalizer, 
                                            plot_truth=False,
                                            plot_func="scatter",
                                            output_filename='NMC111_grid_transfer_comparison_calculated_percentages.pdf',
                                            figsize=(3, 3), legend=False)
    label_list = ["Adaptive {}".format(i + 1) for i in range(pd.read_csv(os.path.join(folders[0], "phase_transition_percentages.csv")).shape[0])]
    analyzer.plot_all_spectra(folders[0], list(range(len(label_list))), label_list, normalizer=normalizer, 
                              plot_measured_data=True,
                              fit_normalizer_with_true_data=False,
                              alpha=0.2,
                              plot_density_estimation=True,
                              output_filename='NMC111_grid_transfer_all_adapt_spectra.pdf',
                              plot_figsize=(6, 3)
                              )
    analyzer.plot_all_spectra(folders[0], list(range(len(label_list))), label_list, normalizer=normalizer, 
                              plot_measured_data=True,
                              fit_normalizer_with_true_data=False,
                              alpha=0.8,
                              plot_density_estimation=False,
                              output_filename='NMC111_grid_transfer_all_adapt_spectra_zoom.pdf',
                              plot_figsize=(3, 3),
                              xlim=(8345, 8355), ylim=(1.4, 1.8),
                              legend=False,
                              axis_labels=False,
                              linewidth=1.5
                              )

