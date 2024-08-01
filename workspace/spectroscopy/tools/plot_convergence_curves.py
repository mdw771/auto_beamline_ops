import os

import matplotlib.pyplot as plt
import numpy as np


flist = [
    "/data/programs/autobl/workspace/spectroscopy/XANES_SLADSNet/outputs/YBCO_raw_randInit/YBCO3data_UnknownExperimentGuide_conv.txt",
    "/data/programs/autobl/workspace/spectroscopy/XANES/outputs/YBCO_raw_randInit/YBCO3data_ComprehensiveAugmentedAcquisitionFunction_gradOrder_2_phiG_0.005970595268105566_phiR_39.514841959638474_phiG2_9.60928119334038e-05_MaternKernel_lengthscale_10.202_nu_2.5_DiscreteOptimizer_conv.txt",
]

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    for f in flist:
        data = np.loadtxt(f)
        ax.plot(data[0, :], data[1, :], label=f)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.set_xlabel('Points measured')
    ax.set_ylabel('RMS')
    # plt.tight_layout()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=1)
    plt.show()
    