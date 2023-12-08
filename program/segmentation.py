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
        self.cell_window_mask = None

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
                self.cell_window_bbox = self.estimate_cell_window_bbox(mask_this_label)
                self.cell_window_mask = self.estimate_cell_window_mask(mask_this_label)
            if not self.should_exclude(mask_this_label):
                self.final_mask += mask_this_label
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.image)
        ax[0].imshow(self.cell_window_mask, alpha=0.5)
        ax[1].imshow(self.image)
        ax[1].imshow(self.final_mask, alpha=0.8)
        plt.show()
        self.run_backsample()

    def estimate_cell_window_bbox(self, mask):
        bbox = self.get_region_bbox(mask)
        window_radius = max(bbox.height, bbox.width) / 2
        if bbox.height < bbox.width:
            bbox.set_sy(int(np.round(bbox.ey - 2 * window_radius)))
        return bbox

    def estimate_cell_window_mask(self, mask):
        """
        Calculate a circular mask given a partial mask.

        :param mask: np.ndarray.
        :return: np.ndarray.
        """
        bbox = self.estimate_cell_window_bbox(mask)
        mask = np.where(mask > 0, 1, 0)
        mask_edge = ndi.binary_dilation(mask, np.ones([3, 3])) - mask
        mask_edge[bbox.sy:int(bbox.sy + 0.7 * bbox.height), bbox.sx:bbox.ex] = 0
        y, x = np.where(mask_edge > 0)
        y = y[::40 // self.downsample]
        x = x[::40 // self.downsample]
        if len(y) < 3:
            raise ValueError('There are not enough pixels for fitting circle. Is the window area too small?')
        yc, xc, r = self.fit_circle(np.stack([y, x], axis=1))
        y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
        circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r ** 2
        return circ_mask


    def should_exclude(self, mask):
        # bbox = self.get_region_bbox(mask)
        # if self.cell_window_bbox.is_isolated_from(bbox):
        if not self.intersects_with_cell_window_mask(mask):
            return True
        if np.count_nonzero(mask) < 100 / self.downsample ** 2:
            return True
        if self.is_round(mask):
            return True
        return False

    def intersects_with_cell_window_mask(self, mask):
        if np.count_nonzero(mask * self.cell_window_mask) > 0:
            return True
        else:
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

    def fit_circle(self, point_list):
        point_list = np.array(point_list)
        y, x = point_list[:, 0], point_list[:, 1]
        a_mat = np.stack([y, x, np.ones_like(y)], axis=1)
        b_vec = y ** 2 + x ** 2
        x_vec = np.linalg.pinv(a_mat) @ b_vec
        yc = x_vec[0] / 2
        xc = x_vec[1] / 2
        r = np.sqrt(x_vec[2] + yc ** 2 + xc ** 2)
        return yc, xc, r

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
