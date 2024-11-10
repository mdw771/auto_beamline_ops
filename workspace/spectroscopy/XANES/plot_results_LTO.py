import os
import glob
import pickle
import contextlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

from plot_results import ResultAnalyzer
import autobl.tools.spectroscopy.xanes as xanestools


if __name__ == '__main__':
    flist = [glob.glob('outputs/LTO_raw_randInit/Sample1*_intermediate_data.pkl')[0],
             glob.glob('outputs/LTO_raw_randInit_noReweighting/Sample1*_intermediate_data.pkl')[0],
             glob.glob('outputs/LTO_raw_randInit_posteriorStddev/Sample1*_intermediate_data.pkl')[0],
             glob.glob('outputs/LTO_raw_randInit_uniformSampling/Sample1*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              'Posterior uncertainty only',
              'Uniform sampling'
              ]
    analyzer = ResultAnalyzer(output_dir='factory')
    analyzer.compare_convergence(flist, labels, output_filename='LTO_comparison_convergence.pdf', auc_range=(0, 30),
                                 ref_line_y=0.004, rms_normalization_factor=1.0)
    analyzer.plot_intermediate(flist[0], interval=6, output_filename='LTO_intermediate.pdf')
    # analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=32,
    #                            zoom_in_range_x=(9010, 9030), zoom_in_range_y=(0.9, 1.2), add_legend=False,
    #                            output_filename='YBCO_intermediate_atNPts_32.pdf')
    # analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=50, add_legend=False,
    #                            normalize_and_detilt=True, normalization_fit_ranges=((8920, 8965), (9040, 9085)),
    #                            output_filename='YBCO_intermediate_atNPts_50_norm_detilt.pdf')
