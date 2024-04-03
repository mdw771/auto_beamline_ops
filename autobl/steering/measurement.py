from typing import Callable, Sequence

import numpy as np
from torch import Tensor
import scipy.interpolate

from autobl.util import *


class Measurement:

    def __init__(self, *args, **kwargs):
        pass

    def measure(self, *args, **kwargs):
        pass


class SimulatedMeasurement(Measurement):

    def __init__(self, f: Callable = None, data: np.ndarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (f is not None) or (data is not None)
        self.f = f
        self.data = data
        self.interp = None
        if self.data is not None:
            if isinstance(self.data, (list, tuple)):
                x_data, y_data = self.data
                x_data = to_numpy(x_data)
                y_data = to_numpy(y_data)
            else:
                x_data = np.arange(len(data))
                y_data = to_numpy(self.data)
            self.interp = lambda pts: scipy.interpolate.interpn(x_data, y_data, pts)

    def measure(self, x, *args, **kwargs):
        """
        Take measurement.

        :param x: Tensor. A list of points where the values are measured. Tensor shape should be [n_pts, n_dims].
        :return:
        """
        if self.f is not None:
            return to_tensor(self.f(x))
        elif self.data is not None:
            x = to_numpy(x)
            x = to_tensor(self.interp(x))
            return x
        else:
            raise ValueError('f or data cannot both be None.')
