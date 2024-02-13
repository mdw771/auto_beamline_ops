import sys
import os
import glob
import argparse

import numpy as np
import tifffile
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import autobl.segmentation


def preprocess(img):
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def test_segmentation_selector(generate_gold=False):
    np.random.seed(196)

    img_cell = tifffile.imread(os.path.join('data', 'saxs_beamline', 'cell_bw_1s_000.tif'))
    img_capillary = tifffile.imread(os.path.join('data', 'saxs_beamline', 'capillaryl_bw_0p015_000.tif'))
    seg_alg_selector = autobl.segmentation.SegmentationAlgorithmSelector(downsample=4)

    seg_alg_selector.set_camera_image(img_cell)
    alg = seg_alg_selector.run_selection()
    assert alg == autobl.segmentation.BubbleSegmentor

    seg_alg_selector.set_camera_image(img_capillary)
    alg = seg_alg_selector.run_selection()
    assert alg == autobl.segmentation.CapillarySegmentor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_segmentation_selector(generate_gold=args.generate_gold)
