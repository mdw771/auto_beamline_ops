from typing import Callable, Optional
import warnings

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
        self.measured_points = torch.tensor([], requires_grad=False)
        self.duplicate_distance_threshold = self.get_duplicate_threshold()

    def maximize(self, acquisition_function: AcquisitionFunction):
        pass

    def get_required_params(self):
        return []

    def get_duplicate_threshold(self):
        assert self.bounds is not None
        perc = 0.001
        duplicate_distance_threshold = (self.bounds[1] - self.bounds[0]) * perc
        duplicate_distance_threshold[torch.isinf(duplicate_distance_threshold)] = 0
        return duplicate_distance_threshold

    def update_sampled_points(self, pts):
        self.measured_points = torch.cat([self.measured_points, pts], dim=0)

    def remove_points_measured_before(self, pts):
        """
        Remove points that have been measured in the past.

        :param pts: Tensor. Tensor of suggested points with shape `[n, d]`.
        :return: Tensor, Tensor. List of points with duplicating points removed, and the mask of points to keep.
        """
        if len(self.measured_points) == 0:
            return pts, torch.ones(len(pts)).bool()
        diff_mat = torch.abs(pts[..., None] - self.measured_points)
        diff_mat = (diff_mat < self.duplicate_distance_threshold).int()
        diff_mat = diff_mat.sum(-1).sum(-1).bool()
        mask = ~diff_mat
        if torch.count_nonzero(mask) == 0:
            warnings.warn('All suggested points have been measured in the past! ({})'.format(pts))
            mask = torch.ones(len(pts)).bool()
        else:
            pts = pts[mask]
        return pts, mask


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
        pts, acq_vals = self.optim_func(acquisition_function, return_best_only=False, **arg_dict)
        pts = pts.reshape([-1, pts.shape[-1]])
        pts, mask = self.remove_points_measured_before(pts)
        acq_vals = acq_vals[mask]

        inds = torch.argsort(acq_vals, descending=True)
        pts = pts[inds]
        acq_vals = acq_vals[inds]

        self.update_sampled_points(pts[0])
        return pts[0:1], acq_vals[0:1]

    def get_required_params(self):
        return self.required_params[self.optim_func]
