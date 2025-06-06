import os
import glob
import pickle
import contextlib
from typing import Optional
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

from plot_results import ResultAnalyzer
import autobl.tools.spectroscopy.xanes as xanestools


logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    flist = [
        glob.glob('outputs/YBCO_raw_uniInit_GPPM/YBCO3data_*_intermediate_data.pkl')[0],
        glob.glob('outputs/YBCO_raw_uniInit_GPPM_noReweighting/YBCO3data_*_intermediate_data.pkl')[0],
        glob.glob('outputs/YBCO_raw_uniInit_GPPM_posteriorStddev/YBCO3data_*_intermediate_data.pkl')[0],
        glob.glob('outputs/YBCO_raw_uniInit_GPPM_UCB_kappa_12/YBCO3data_*_intermediate_data.pkl')[0],
        glob.glob('outputs/YBCO_raw_uniInit_GPPM_EI/YBCO3data_*_intermediate_data.pkl')[0],
        glob.glob('outputs/YBCO_raw_uniInit_GPPM_uniformSampling/YBCO3data_*_intermediate_data.pkl')[0]
    ]
    labels = ['Comprehensive acq. + reweighting',
              'Comprehensive acq.',
              "Posterior uncertainty only",
              "Upper confidence bound",
              "Expected improvement",
              'Uniform sampling'
              ]
    
    normalizer = xanestools.XANESNormalizer()
    normalizer.load_state("outputs/YBCO_raw_uniInit_GPPM/normalizer_state.npy")
    print(normalizer.__dict__)
    
    analyzer = ResultAnalyzer(output_dir='factory')
    analyzer.compare_convergence(flist, labels, output_filename='YBCO_comparison_convergence.pdf', auc_range=(0, 40), 
                                 ref_line_y=0.005, rms_normalization_factor=1.0, normalizer=normalizer, interpolate_on_fine_grid=True)
    # analyzer.plot_auc_multipass(
    #     flist,
    #     labels,
    #     auc_range=(0, 30), 
    #     normalizer=normalizer, 
    #     output_filename='YBCO_auc_multipass.pdf'
    # )
    analyzer.plot_intermediate(flist[0], interval=12, n_cols=1, sharex=True, iter_list=(0, 15, 30), figsize=(8, 6), output_filename='YBCO_intermediate.pdf')
    analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=30,
                               zoom_in_range_x=(9002, 9008), zoom_in_range_y=(1.1, 1.3), add_legend=False,
                               output_filename='YBCO_intermediate_atNPts_30.pdf')
    # analyzer.compare_estimates(flist[0::1], labels[0::1], at_n_pts=40, add_legend=False,
    #                            normalizer=normalizer,
    #                            output_filename='YBCO_intermediate_atNPts_40_norm_detilt.pdf')
    analyzer.compare_intermediate(
        flist[:2], 
        labels[:2], 
        interval=5, 
        n_cols=2, 
        add_legend=False, 
        sharex=True, 
        figsize=(10, 12), 
        output_filename='YBCO_comparison_intermediate.pdf'
    )
    analyzer.compare_convergence(
        [
            glob.glob('outputs/YBCO_raw_uniInit_GPPM_addedNoiseVar_1.0e-04_noiseVar_1.0e-04/YBCO3data_*_intermediate_data.pkl')[0],
            glob.glob('outputs/YBCO_raw_uniInit_GPPM_addedNoiseVar_1.0e-04_noiseVar_1.0e-02/YBCO3data_*_intermediate_data.pkl')[0],
            # glob.glob('outputs/YBCO_raw_uniInit_GPPM_addedNoiseVar_1.0e-04_noiseVar_1.0e-03/YBCO3data_*_intermediate_data.pkl')[0],
            # glob.glob('outputs/YBCO_raw_uniInit_GPPM_addedNoiseVar_1.0e-04_noiseVar_1.0e-05/YBCO3data_*_intermediate_data.pkl')[0],
            glob.glob('outputs/YBCO_raw_uniInit_GPPM_addedNoiseVar_1.0e-04_noiseVar_1.0e-06/YBCO3data_*_intermediate_data.pkl')[0],
        ], 
        [
            '$10^{-4}$',
            '$10^{-2}$',
            # '$10^{-3}$',
            # '$10^{-5}$',
            '$10^{-6}$',
        ], 
        output_filename='YBCO_comparison_convergence_noise.pdf', 
        auc_range=(0, 30), 
        ref_line_y=None, 
        rms_normalization_factor=1.0, 
        normalizer=normalizer, 
        interpolate_on_fine_grid=True
    )