"""Image processing including reconstruction, fitting"""

from typing import Literal, Optional, Tuple

# import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from autobl.bounding_box import BoundingBox


def fit_circle(point_list, return_residue=False):
    """
    Solve a least squares system to fit points to a circle defined by center
    (yc, xc) and radius r

    A = [ |  |  |  ]
        [ y, x,ones]
        [ |  |  |  ]
    b = [      |     ]
        [ y**2 + x**2]
        [      |     ]
    x = [2*yc]
        [2*xc]
        [r**2 - yc**2 - xc**2]

    s.t. Ax = [2*y*yc + 2*x*xc + r**2 - yc**2 - xc**2] == b = [y**2 + x**2]

    Derived from:
       (y - yc)**2           + (x - xc)**2           = r**2
       y**2 - 2*y*yc + yc**2 + x**2 - 2*x*yc + xc**2 = r**2
    """
    point_list = np.array(point_list)
    y, x = point_list[:, 0], point_list[:, 1]
    a_mat = np.stack([y, x, np.ones_like(y)], axis=1)
    b_vec = y**2 + x**2
    x_vec = np.linalg.pinv(a_mat) @ b_vec
    residue = np.mean((a_mat @ x_vec - b_vec) ** 2)
    yc = x_vec[0] / 2
    xc = x_vec[1] / 2
    r = np.sqrt(x_vec[2] + yc**2 + xc**2)
    if return_residue:
        return (yc, xc, r), residue
    return yc, xc, r


def calculate_circle_fitting_residue(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[: ref_mask.shape[0], : ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r**2
    residue = np.mean((circ_mask - ref_mask) ** 2)

    return residue


def calculate_circle_fitting_iou(circle_params, ref_mask):
    """
    Calculate the MSE residue between a disk determined by (yc, xc, r) and a binary mask.
    """
    yc, xc, r = circle_params
    y, x = np.mgrid[: ref_mask.shape[0], : ref_mask.shape[1]]
    circ_mask = (y - yc) ** 2 + (x - xc) ** 2 <= r**2
    area_intersect = np.sum(circ_mask * ref_mask)
    iou = area_intersect / (np.sum(circ_mask) + np.sum(ref_mask) - area_intersect)
    return iou


def get_region_bbox(mask):
    """
    Get the bounding box of a region
    """
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
        pos_count[i, 1] = np.count_nonzero(
            np.logical_and(peak_list >= v1, peak_list <= v2)
        )
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
    d = np.abs(
        np.cross(line_pt_2 - line_pt_1, pts - line_pt_1)
        / np.linalg.norm(line_pt_2 - line_pt_1)
    )
    return d


class DenseReconstructor:
    """
    Reconstruct an image in dense pixel grid from sparse points.

    Inverse distance weighted interpolation (IDW) is adapted from FaST:
    https://github.com/saugatkandel/fast_smart_scanning
    """

    def __init__(
        self, method: Literal["idw", "linear"] = "idw", options: Optional[dict] = None
    ):
        """
        The constructor.

        :param method: str. The method of interpolation.
            'idw': inverse distance weighted interpolation.
            'linear': linear interpolation using scipy.interpolate.griddata.
        :param options: Optional[dict]. Options.
            When method is 'idw':
                n_neighbors: number of neighbors whose values are used to
                    calculate the interpolation for each point.
        :return: np.ndarray.
        """
        self.method = method
        self.options = options if options is not None else {}

    def reconstruct(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray],
    ):
        """
        Reconstruct a dense image.

        :param points: np.ndarray. A (N, 2) array of measured points.
        :param values: np.ndarray. A 1-D array of measured values.
        :param meshgrids:
        :return:
        """
        if self.method == "linear":
            recon = self.reconstruct_linear(points, values, meshgrids)
        elif self.method == "idw":
            recon = self.reconstruct_idw(points, values, meshgrids)
        else:
            raise ValueError(f"{self.method} is not a valid method.")
        return recon

    def reconstruct_linear(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray],
    ):
        """
        Linear interpolation (using simplices) of points and values
        """
        grid_y, grid_x = meshgrids
        recon = scipy.interpolate.griddata(points, values, (grid_y, grid_x))
        return recon

    def reconstruct_idw(
        self,
        points: np.ndarray,
        values: np.ndarray,
        meshgrids: Tuple[np.ndarray, np.ndarray],
    ):
        """
        Inverse distance weighted interpolation

        :param points: np.ndarray. points to interpolate between, shape (num
            interp points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num
            interp points,)
        :param meshgrids: Tuple[np.ndarray, np.ndarray]. grid to get
            interpolated values for
        """
        # grid_y, grid_x = meshgrids
        n_neighbors = (
            4
            if "n_neighbors" not in self.options.keys()
            else self.options["n_neighbors"]
        )
        knn_engine = NearestNeighbors(n_neighbors=n_neighbors)
        knn_engine.fit(points)

        # Find nearest measured points for each queried point.
        queried_points = np.stack(meshgrids, axis=-1).reshape(-1, len(meshgrids))
        nn_dists, nn_inds = knn_engine.kneighbors(queried_points)
        nn_weights = self._compute_neighbor_weights(nn_dists)
        nn_values = np.take(values, nn_inds)

        recon = np.sum(nn_values * nn_weights, axis=1)
        recon = recon.reshape(meshgrids[0].shape)
        return recon

    @staticmethod
    def _compute_neighbor_weights(neighbor_distances, power=2, eps=1e-7):
        """
        Calculating the weights for how each neighboring data point contributes
        to the reconstruction for the current location.

        First, the weights are calculated to be inversely proportional to the
        distance from teh current point. Next, the weights are normalized so
        that the total weight sums up to 1 for each reconstruction point.
        """
        unnormalized_weights = 1.0 / (np.power(neighbor_distances, power) + eps)
        sum_over_row = np.sum(unnormalized_weights, axis=1, keepdims=True)
        weights = unnormalized_weights / sum_over_row
        return weights

    @staticmethod
    def _idw(
        points: np.ndarray, values: np.ndarray, xi: np.ndarray, power: float = 2.0
    ) -> np.ndarray:
        """Inverse distance weighted interpolation

        Interpolate by weighting by inverse distance

        :param points: np.ndarray. points to interpolate between, shape (num interp
            points, dimension)
        :param values: np.ndarray. values to interpolate between, shape (num interp
            points,)
        :param xi: np.ndarray. points to get interpolated values at, shape (num
            sample points, dimension)
        :return: np.ndarray. interpolated values, shape (num sample points,)
        """
        # Get inverse distances from sample points -> interp points
        inv_distances = 1 / (cdist(xi, points) ** power)
        # Get sum of inverse distances for each sample
        sum_inv_distances = np.sum(inv_distances, axis=1)
        # For sample points == interp points...
        mask = ~np.isfinite(inv_distances)
        # ...set some values to 0.0...
        inv_distances[~np.isfinite(sum_inv_distances), :] = 0.0
        # ...but make sure the right values are non-zero
        inv_distances[mask] = 1.0
        sum_inv_distances[~np.isfinite(sum_inv_distances)] = 1.0
        # Multiply values by inverse distances and divide by sum
        vi = np.sum(values[np.newaxis, :] * inv_distances, axis=1) / sum_inv_distances
        return vi
