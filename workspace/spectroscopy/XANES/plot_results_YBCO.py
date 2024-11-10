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
    flist = [glob.glob('outputs/YBCO_raw_randInit/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/YBCO_raw_randInit_noReweighting/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/YBCO_raw_randInit_posteriorStddev/YBCO3data_*_intermediate_data.pkl')[0],
             glob.glob('outputs/YBCO_raw_randInit_uniformSampling/YBCO3data_*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              'Posterior uncertainty only',
              'Uniform sampling'
              ]
    
    normalizer = xanestools.XANESNormalizer()
    normalizer.load_state("outputs/YBCO_raw_randInit/normalizer_state.npy")
    print(normalizer.__dict__)
    
    analyzer = ResultAnalyzer(output_dir='factory')
    analyzer.compare_convergence(flist, labels, output_filename='YBCO_comparison_convergence.pdf', auc_range=(0, 30), 
                                 ref_line_y=0.005, rms_normalization_factor=1.0, normalizer=normalizer)
    analyzer.plot_intermediate(flist[0], interval=6, output_filename='YBCO_intermediate.pdf')
    analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=30,
                               zoom_in_range_x=(9007, 9017), zoom_in_range_y=(1.075, 1.16), add_legend=False,
                               output_filename='YBCO_intermediate_atNPts_30.pdf')
    analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=50, add_legend=False,
                               normalizer=normalizer,
                               output_filename='YBCO_intermediate_atNPts_50_norm_detilt.pdf')