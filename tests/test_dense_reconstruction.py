import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile

import autobl.image_proc


def test_dense_reconstruction(generate_gold=False, debug=False):
    np.random.seed(123)

    img = tifffile.imread('data/xrf/xrf_2idd_Cs_L.tiff')
    n_pts = 10000
    points_y = np.random.randint(0, img.shape[0], n_pts)
    points_x = np.random.randint(0, img.shape[1], n_pts)
    points = np.stack([points_y, points_x], axis=-1)
    values = img[points_y, points_x]

    reconstructor = autobl.image_proc.DenseReconstructor(method='idw', options={'n_neighbors': 10})
    recon = reconstructor.reconstruct(points, values, np.mgrid[:img.shape[0], :img.shape[1]])

    if debug:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(recon)
        plt.show()
    gold_dir = os.path.join('gold_data', 'test_dense_reconstruction')
    if not os.path.exists(gold_dir):
        os.makedirs(gold_dir)
    if generate_gold:
        tifffile.imwrite(os.path.join(gold_dir, 'interpolated.tiff'), recon)
    else:
        recon_ref = tifffile.imread(os.path.join(gold_dir, 'interpolated.tiff'))
        assert np.allclose(recon, recon_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_dense_reconstruction(generate_gold=args.generate_gold, debug=True)
