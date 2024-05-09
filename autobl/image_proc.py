import numpy as np

from autobl.bounding_box import BoundingBox


def fit_circle(point_list, return_residue=False):
    point_list = np.array(point_list)
    y, x = point_list[:, 0], point_list[:, 1]
    a_mat = np.stack([y, x, np.ones_like(y)], axis=1)
    b_vec = y ** 2 + x ** 2
    x_vec = np.linalg.pinv(a_mat) @ b_vec
    residue = np.mean((a_mat @ x_vec - b_vec) ** 2)
    yc = x_vec[0] / 2
    xc = x_vec[1] / 2
    r = np.sqrt(x_vec[2] + yc ** 2 + xc ** 2)
    if return_residue:
        return (yc, xc, r), residue
    return yc, xc, r


def calculate_circle_fitting_residue(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[:ref_mask.shape[0], :ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r ** 2
    residue = np.mean((circ_mask - ref_mask) ** 2)
    import matplotlib.pyplot as plt
    return residue

def calculate_circle_fitting_iou(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[:ref_mask.shape[0], :ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r ** 2
    area_intersect = np.sum(circ_mask * ref_mask)
    iou = area_intersect / (np.sum(circ_mask) + np.sum(ref_mask) - area_intersect)
    return iou


def get_region_bbox(mask):
    ys, xs = np.nonzero(mask)
    return BoundingBox([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1])


def find_window_location_with_most_peaks(window_size, peak_list):
    """
    Find the location of a window that contains the most peaks.

    :param window_size: int.
    :param peak_list: list[int]. Peak locations in pixel.
    :param range: tuple(int, int). Allowed range of the starting point of the window.
    :return: int.
    """
    peak_list = np.sort(peak_list)
    st = peak_list[0]
    end = peak_list[-1] - window_size
    if end <= st:
        return st
    pos_count = np.zeros([end - st + 1, 2])
    for i, x in enumerate(range(st, end + 1)):
        v1 = st
        v2 = st + window_size
        pos_count[i, 0] = x
        pos_count[i, 1] = np.count_nonzero(np.logical_and(peak_list >= v1, peak_list <= v2))
    i_max = np.argmax(pos_count[:, 1])
    return int(pos_count[i_max, 0])


def point_to_line_distance(pts, line_pt_1, line_pt_2):
    """
    Find the perpendicular distance from an array of points to a vector.

    :param pts: np.ndarray. Array of point.
    :param line_pt_1: np.ndarray. The first point on the line.
    :param line_pt_2: np.ndarray. The second point on the line.
    :return: A [pts.shape[0],] array.
    """
    d = np.abs(np.cross(line_pt_2 - line_pt_1, pts - line_pt_1) / np.linalg.norm(line_pt_2 - line_pt_1))
    return d
