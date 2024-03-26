from typing import Callable

import numpy as np


class Measurement:

    def __init__(self, *args, **kwargs):
        pass

    def measure(self, *args, **kwargs):
        pass


class SimulatedMeasurement(Measurement):

    def __init__(self, f: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    def measure(self, x, *args, **kwargs):
        return self.f(x)
