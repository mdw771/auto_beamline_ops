import os
import argparse

import numpy as np
import tifffile
from skimage.color import rgb2gray

import autobl.analysis.segmentation


def preprocess(img):
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def test_capillary_segmentation(generate_gold=False):
    np.random.seed(196)

    img = tifffile.imread('data/saxs_beamline/capillaryl_bw_0p015_000.tif')
    img = preprocess(img)

    segmentor = autobl.analysis.segmentation.CapillarySegmentor(downsample=4)
    segmentor.set_camera_image(img)
    scannable_mask = segmentor.run(return_original_scale=False)

    test_name = os.path.splitext(os.path.basename(__file__))[0]
    gold_dir = os.path.join('gold_data', test_name)

    if generate_gold:
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
        tifffile.imwrite(os.path.join(gold_dir, 'segment_mask.tiff'), scannable_mask)
    else:
        gold_img = tifffile.imread(os.path.join(gold_dir, 'segment_mask.tiff'))
        assert np.allclose(gold_img, scannable_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_capillary_segmentation(generate_gold=args.generate_gold)
