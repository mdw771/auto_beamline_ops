import os
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from autobl.util import *


if __name__ == '__main__':
    flist = ['outputs/auto_weight/YBCO3data_ComprehensiveAugmentedAcquisitionFunction_gradOrder_2_phiG_0.0025943329162371733_phiR_351.1701040862937_phiG2_7.492389054147616e-05_MaternKernel_lengthscale_7.000_nu_2.5_DiscreteOptimizer_intermediate_data.pkl',
             'outputs/auto_weight_posterior_stddev/YBCO3data_PosteriorStandardDeviation_MaternKernel_lengthscale_7.000_nu_2.5_DiscreteOptimizer_intermediate_data.pkl'
             ]
    labels = ['Comprehensive acquisition',
              'Posterior standard deviation only']
    rms_all_files = []
    n_pts_all_files = []
    for f in flist:
        rms_list = []
        n_pts = []
        data = pickle.load(open(f, 'rb'))
        data_true = data['data_y']
        for i, data_estimated in enumerate(data['mu_list']):
            r = rms(data_estimated, data_true)
            rms_list.append(r)
            n_pts.append(data['n_measured_list'][i])
        rms_all_files.append(rms_list)
        n_pts_all_files.append(n_pts)

    fig, ax = plt.subplots(1, 1)
    for i in range(len(rms_all_files)):
        ax.plot(n_pts_all_files[i], rms_all_files[i], label=labels[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.set_xlabel('Points measured')
    ax.set_ylabel('RMS')
    plt.tight_layout()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=1)
    plt.savefig('outputs/compare_YBCO_comprehensive_vs_posteriorStd.pdf', bbox_inches='tight')
