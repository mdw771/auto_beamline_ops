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
    flist = [glob.glob('outputs/Pt_raw_uniInit_GPPM_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_uniInit_GPPM_noReweighting_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_uniInit_GPPM_posteriorStddev_bgGrad/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_uniInit_GPPM_UCB_kappa_5/Pt_*_intermediate_data.pkl')[0],
             glob.glob('outputs/Pt_raw_uniInit_GPPM_uniformSampling_bgGrad/Pt_*_intermediate_data.pkl')[0]
             ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              'Posterior uncertainty only',
              'Upper confidence bound',
              'Uniform sampling'
              ]

    normalizer = xanestools.XANESNormalizer()
    normalizer.load_state("outputs/Pt_raw_uniInit_GPPM_bgGrad/normalizer_state.npy")

    analyzer = ResultAnalyzer(output_dir='factory')
    
    analyzer.style_list = analyzer.style_list[:4] + analyzer.style_list[5:]
    analyzer.color_list = analyzer.color_list[:4] + analyzer.color_list[5:]
    
    # analyzer.plot_intermediate(flist[0], interval=1, make_animation=True, output_filename='Pt_intermediate_animation.mp4')
    # analyzer.plot_intermediate(flist[3], interval=1, make_animation=True, output_filename='Pt_intermediate_uniform_sampling_animation.mp4')
    analyzer.compare_convergence(
        flist, labels, ref_line_y=0.005, add_legend=False, output_filename='Pt_comparison_convergence.pdf', 
        figsize=(8, 4), auc_range=(0, 50), normalizer=normalizer, #indices_to_skip=(15,)
    )
    analyzer.compare_convergence(
        flist, labels, ref_line_y=0.005, add_legend=True, output_filename='Pt_comparison_convergence_legend.pdf', 
        figsize=(8, 4), auc_range=(0, 50), normalizer=normalizer, #indices_to_skip=(15,)
    )
    # analyzer.plot_auc_multipass(flist, labels, auc_range=(0, 35), normalizer=normalizer, output_filename='Pt_auc_multipass.pdf')
    analyzer.plot_intermediate(flist[0], iter_list=(0, 20, 40), n_cols=1, sharex=True, figsize=(8, 8), output_filename='Pt_intermediate.pdf')
    analyzer.compare_intermediate(
        flist[0:2] + flist[3:4], labels, interval=10, n_cols=2, add_legend=False, sharex=True, figsize=(10, 8), output_filename='Pt_comparison_intermediate.pdf',
        marker_size=15,
        color_list=analyzer.color_list[0:2] + analyzer.color_list[3:4],
        style_list=analyzer.style_list[0:2] + analyzer.style_list[3:4],
        marker_style_list=analyzer.marker_style_list[0:2] + analyzer.marker_style_list[3:4],
    )
    analyzer.compare_estimates(
        flist[0::1], labels[0::1], at_n_pts=60,
        zoom_in_range_x=(11567, 11570), zoom_in_range_y=(0.47, 0.49),
        output_filename='Pt_intermediate_atNPts_60.pdf',
    )
