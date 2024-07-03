import argparse
import os

import numpy as np

import autobl.tools.spectroscopy.xanes as xanestools
from autobl.steering.io_util import *

    

def test_xanes_normalizer(generate_gold=False, debug=False):
    dataset = NORSpectroscopyDataset('data/xanes', file_pattern='10Feb_PtL3_025_042C.xmu', data_column='xmu')
    data = dataset[0]
    energies = dataset.energies_ev
    
    normalizer = xanestools.XANESNormalizer(fit_ranges=((11400, 11500), (11650, 11850)), normalization_order=2)
    normalizer.fit(energies, data)
    data_normalized = normalizer.apply(energies, data)
    
    if debug:
        print('Edge loc: {}'.format(normalizer.edge_loc))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(energies, data)
        ax[0].plot(energies, np.poly1d(normalizer.p_pre)(energies))
        ax[0].plot(energies, np.poly1d(normalizer.p_post)(energies))
        ax[0].grid()
        ax[1].plot(energies, data_normalized)
        ax[1].grid()
        plt.show()

    gold_dir = "gold_data/test_xanes_normalizer"
    if generate_gold:
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
            np.save(os.path.join(gold_dir, "normalized_data.npy"), data_normalized)
    else:
        data_gold = np.load(os.path.join(gold_dir, "normalized_data.npy"), allow_pickle=True)
        assert np.allclose(data_gold, data_normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()
    
    test_xanes_normalizer(generate_gold=args.generate_gold, debug=True)
    