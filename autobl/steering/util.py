import torch
import numpy as np


def estimate_noise_std(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the noise standard deviation of the given 1D data.
    The data is assumed to only have a linear background, which will
    be fit and subtracted. The noise standard deviation is then taken
    as the standard deviation of the residuals.
    
    Parameters
    ----------
    x : np.ndarray
        The x-values of the data.
    y : np.ndarray
        The y-values of the data.

    Returns
    -------
    noise_std : float
        The estimated noise standard deviation.
    """
    # Fit a linear background
    fit = np.polyfit(x, y, 1)
    background = np.polyval(fit, x)
    residuals = y - background
    noise_std = np.std(residuals)
    return noise_std
