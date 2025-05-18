import torch
import numpy as np


def estimate_noise_variance(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the noise variance of the given 1D data.
    The data is assumed to only have a linear background, which will
    be fit and subtracted. The noise variance is then taken
    as the variance of the residuals.
    
    Parameters
    ----------
    x : np.ndarray
        The x-values of the data.
    y : np.ndarray
        The y-values of the data.

    Returns
    -------
    noise_variance : float
        The estimated noise variance.
    """
    # Fit a linear background
    fit = np.polyfit(x, y, 1)
    background = np.polyval(fit, x)
    residuals = y - background
    noise_variance = np.var(residuals)
    return noise_variance
