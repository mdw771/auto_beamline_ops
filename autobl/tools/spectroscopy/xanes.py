from typing import Optional, Tuple, Sequence

import numpy as np
import scipy


def estimate_edge_location_and_width(data_x: np.ndarray, data_y: np.ndarray, x_dense: Optional[np.ndarray] = None,
                                     return_in_pixel_unit=False) -> Tuple[float | int, float]:
    """
    Estimate the location and width of the absorption edge in XANES.

    :param data_x: ndarray. Energies in eV or normalized unit.
    :param data_y: ndarray.
    :param x_dense: Optional[ndarray]. A dense array of energies to interpolate data on.
    :param return_in_pixel_unit: bool. If True, the unit of results will be pixel.
    :return: peak location and width in the same unit as data_x.
    """
    # Interpolate a dense spectrum.
    if x_dense is None:
        x_dense = np.linspace(data_x[0], data_x[-1], len(data_x) * 10)
    y_dense = scipy.interpolate.CubicSpline(data_x, data_y)(x_dense)

    # Calculate gradient in the dense spectrum.
    grad_y = (y_dense[2:] - y_dense[:-2]) / (x_dense[2:] - x_dense[:-2])
    dense_psize = x_dense[1] - x_dense[0]

    # Find gradient peaks.
    peak_locs, peak_properties = scipy.signal.find_peaks(grad_y, height=grad_y.max() * 0.5, width=1)
    edge_ind = np.argmax(peak_properties['peak_heights'])
    peak_loc = peak_locs[edge_ind]
    peak_width = peak_properties['widths'][edge_ind]

    # Convert peak location and width from pixels to desired unit.
    if not return_in_pixel_unit:
        peak_loc = x_dense[peak_loc]
        peak_width = peak_width * dense_psize
    return peak_loc, peak_width


def detilt_and_normalize(data_x, data_y, fit_ranges=None, return_fits=False, fits_to_apply=None):
    """
    Remove the background slope and normalize a XANES spectrum using its pre-edge and post-edge values.

    :param data_x: ndarray. Energies in eV.
    :param data_y: ndarray.
    :return: ndarray. Processed spectrum.
    """
    if fit_ranges is None:
        edge_loc, edge_width = estimate_edge_location_and_width(data_x, data_y)
        range_pre = (data_x[0], edge_loc - edge_width * 3)
        range_post = (edge_loc + edge_width * 3, data_x[-1])
    else:
        range_pre, range_post = fit_ranges

    if fits_to_apply is None:
        # Fit pre-edge
        mask = (data_x < range_pre[1]) & (data_x >= range_pre[0])
        x = data_x[mask]
        y = data_y[mask]
        p_pre = np.polyfit(x, y, 1)

        # Fit post-edge
        mask = (data_x > range_post[0]) & (data_x <= range_post[1])
        x = data_x[mask]
        y = data_y[mask]
        # p_post = np.polyfit(x, y, 1)
        p_post = np.array([p_pre[0], np.mean(y - x * p_pre[0])])
    else:
        p_pre, p_post = fits_to_apply

    data_corrected = data_y - data_x * p_pre[0] - p_pre[1]
    edge_height = (p_post[1] - p_pre[1]) / np.sqrt(p_pre[0] ** 2 + 1)
    data_corrected = data_corrected / edge_height

    if return_fits:
        return data_corrected, (p_pre, p_post)
    return data_corrected
