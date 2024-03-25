import numpy as np


class Simulator:

    def __init__(self, *args, **kwargs):
        pass


class DummySimulator(Simulator):

    def __init__(self, measurement_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement_dims = measurement_dims

    def forward(self, *args, **kwargs):
        return np.random.rand(self.measurement_dims)
