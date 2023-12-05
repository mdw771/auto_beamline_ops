import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.transform
import skimage.filters
import sklearn.decomposition
import tqdm

from bounding_box import BoundingBox


class BubbleSegmentor:

    def __init__(self, downsample=4):
        self.image = None
        self.downsample = downsample
        self.orig_shape = None
        self.final_mask = None
        self.cell_window_bbox = None

    def set_camera_image(self, img):
        self.image = img

    def run(self):
        self.run_downsample()
        image_blur = ndi.gaussian_filter(self.image, 2 / self.downsample)
        thresh = skimage.filters.threshold_otsu(image_blur)
        thresh_mask = (self.image > thresh).astype(int)
        ndi.binary_closing(thresh_mask, structure=np.ones([5, 5]))
        self.final_mask = np.zeros_like(thresh_mask)
        labeled_thresh_mask, n_labels = ndi.label(thresh_mask)
        labels_sorted = self.sort_labels_by_area(labeled_thresh_mask, n_labels)

        for i, label in enumerate(tqdm.tqdm(labels_sorted)):
            mask_this_label = labeled_thresh_mask == label
            if i == 0:
                self.final_mask += mask_this_label
                self.cell_window_bbox = self.get_region_bbox(mask_this_label)
                continue
            if not self.should_exclude(mask_this_label):
                self.final_mask += mask_this_label
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.image)
        ax[1].imshow(self.final_mask)
        plt.show()
        self.run_backsample()

    def should_exclude(self, mask):
        bbox = self.get_region_bbox(mask)
        if not self.cell_window_bbox.contains(bbox):
            return True
        if np.count_nonzero(mask) < 100 / self.downsample ** 2:
            return True
        if self.is_round(mask):
            return True
        return False

    @staticmethod
    def get_region_bbox(mask):
        ys, xs = np.nonzero(mask)
        return BoundingBox([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])

    def is_round(self, mask):
        """
        Calculate the eigenvalues of the principle axes of the non-zero elements in a given mask, then calculate
        the ratio. If the ratio is close enough to 1, the distribution can be considered to be close enough to a disk.

        :param mask: np.ndarray.
        :return: bool.
        """
        pca = sklearn.decomposition.PCA(n_components=2)
        y_inds, x_inds = np.nonzero(mask)
        x = np.stack([y_inds, x_inds], axis=-1)
        res = pca.fit(x)
        eigen_value_ratio = res.explained_variance_ratio_[0] / res.explained_variance_ratio_[1]
        if eigen_value_ratio < 10:
            return True
        else:
            return False
        plt.figure()
        plt.imshow(mask)
        plt.show()

    def sort_labels_by_area(self, labeled_mask, n_labels=None):
        if n_labels is None:
            n_labels = labeled_mask.max()
        res = []
        for label in range(1, n_labels + 1):
            res.append([label, np.count_nonzero(labeled_mask == label)])
        res = np.array(res)
        res = res[np.argsort(res[:, 1])[::-1]]
        return res[:, 0].astype(int)

    def run_downsample(self):
        self.orig_shape = self.image.shape
        if self.downsample == 1:
            return
        self.image = skimage.transform.resize(self.image, [x // self.downsample for x in self.image.shape], order=1)

    def run_backsample(self):
        if self.downsample == 1:
            return
        self.final_mask = skimage.transform.resize(self.final_mask, self.orig_shape, order=1)
        self.final_mask = np.where(self.final_mask > 0.5, 1, 0)
