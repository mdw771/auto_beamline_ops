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
    flist = [glob.glob('outputs/Pt_raw_qrandInit_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_qrandInit_noReweighting_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_qrandInit_posteriorStddev_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_qrandInit_uniformSampling_bgGrad/Pt_*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              'Posterior uncertainty only',
              'Uniform sampling'
              ]

    normalizer = xanestools.XANESNormalizer()
    normalizer.load_state("outputs/Pt_raw_qrandInit_bgGrad/normalizer_state.npy")

    analyzer = ResultAnalyzer(output_dir='factory')
    # analyzer.plot_intermediate(flist[0], interval=1, make_animation=True, output_filename='Pt_intermediate_animation.mp4')
    # analyzer.plot_intermediate(flist[3], interval=1, make_animation=True, output_filename='Pt_intermediate_uniform_sampling_animation.mp4')
    analyzer.compare_convergence(flist, labels, ref_line_y=0.005, add_legend=False, output_filename='Pt_comparison_convergence.pdf', figsize=(8, 4), auc_range=(0, 35), normalizer=normalizer)
    analyzer.plot_intermediate(flist[0], iter_list=(0, 5, 10, 30), n_cols=1, sharex=True, figsize=(8, 8), output_filename='Pt_intermediate.pdf')
    analyzer.compare_intermediate(flist, labels, interval=5, n_cols=2, add_legend=False, sharex=True, figsize=(10, 12), output_filename='Pt_comparison_intermediate.pdf')
    # analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=32,
    #                            zoom_in_range_x=(9010, 9030), zoom_in_range_y=(0.7, 1.1),
    #                            output_filename='Pt_intermediate_atNPts_32.pdf')