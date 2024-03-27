from typing import Callable, Optional

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import *


class Optimizer:
    """
    Acquisition function optimizer.
    """
    def __init__(
            self,
            *args,
            bounds: Optional[Tensor] = None,
            num_candidates: int = 1,
            **kwargs
    ) -> None:
        self.bounds = bounds
        self.num_candidates = num_candidates

    def maximize(self, acquisition_function: AcquisitionFunction):
        pass

    def get_required_params(self):
        return []


class BoTorchOptimizer(Optimizer):
    """
    Wrapper of BoTorch optimization functions.
    """
    default_params = {
        optimize_acqf: {
            "num_restarts": 5,
            "raw_samples": 10
        }
    }

    required_params = {
        optimize_acqf: ['bounds', 'q', 'num_restarts']
    }

    def __init__(
            self,
            *args,
            bounds: Optional[Tensor] = None,
            num_candidates: int = 1,
            optim_func: Callable = optimize_acqf,
            optim_func_params: Optional[dict] = None,
            **kwargs
    ) -> None:
        """
        The constructor.

        :param bounds: Optional[Tensor[Tensor[float, ...], Tensor[float, ...]]]. The lower and upper bound of the
                       search space.
        :param num_candidates: int. The number of candidates to suggest. When working with analytical acquisition
                               functions, this must be 1.
        :param optim_func: Callable. An optimization function from botorch.optim.optimize, such as optimize_acqf.
        :param optim_func_params: Optional[dict]. The parameters of the BoTorch optimization function other than
                                  `acq_function`, `bounds`, and `q`.
        """
        super().__init__(*args, bounds=bounds, num_candidates=num_candidates, **kwargs)
        self.optim_func = optim_func
        self.optim_func_params = optim_func_params
        if self.optim_func_params is None:
            self.optim_func_params = {}

    def get_argument_dict(self):
        arg_dict = {**self.optim_func_params}
        for arg in self.default_params[self.optim_func].keys():
            if arg not in arg_dict.keys():
                arg_dict[arg] = self.default_params[self.optim_func][arg]

        if 'bounds' in self.required_params[self.optim_func] and 'bounds' not in arg_dict.keys():
            arg_dict['bounds'] = self.bounds
        if 'q' in self.required_params[self.optim_func] and 'q' not in arg_dict.keys():
            arg_dict['q'] = self.num_candidates

        for arg in self.required_params[self.optim_func]:
            if arg not in arg_dict.keys():
                raise ValueError("{} is required by {}, but is not provided.".format(arg, self.optim_func.__name__))
        return arg_dict

    def maximize(self, acquisition_function: AcquisitionFunction):
        """
        Maximize the acquisition function.

        :param acquisition_function: AcquisitionFunction. The acquisition function to optimize.
        :return: Tensor, Tensor. The locations and values of the optima.
        """
        arg_dict = self.get_argument_dict()
        return self.optim_func(acquisition_function, **arg_dict)

    def get_required_params(self):
        return self.required_params[self.optim_func]
