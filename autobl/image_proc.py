import numpy as np

from autobl.bounding_box import BoundingBox


def fit_circle(point_list):
    point_list = np.array(point_list)
    y, x = point_list[:, 0], point_list[:, 1]
    a_mat = np.stack([y, x, np.ones_like(y)], axis=1)
    b_vec = y ** 2 + x ** 2
    x_vec = np.linalg.pinv(a_mat) @ b_vec
    yc = x_vec[0] / 2
    xc = x_vec[1] / 2
    r = np.sqrt(x_vec[2] + yc ** 2 + xc ** 2)
    return yc, xc, r


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
