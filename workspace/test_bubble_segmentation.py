import sys
sys.path.insert(0, '/data/programs/auto_alignment_bubble_segmentation/program')
import os

import tifffile

import segmentation


if __name__ == '__main__':
    img = tifffile.imread('data/raw/cell_bw_1s_000.tif')
    segmentor = segmentation.BubbleSegmentor(downsample=4)
    segmentor.set_camera_image(img)
    segmentor.run()
