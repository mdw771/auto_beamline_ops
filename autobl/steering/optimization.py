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
        perc = 1e-3
        duplicate_distance_threshold = (self.bounds[1] - self.bounds[0]) * perc
        duplicate_distance_threshold[torch.isinf(duplicate_distance_threshold)] = 0
        return duplicate_distance_threshold

    def update_sampled_points(self, pts):
        self.measured_points = torch.cat([self.measured_points, pts], dim=0)

    def find_duplicate_point_mask(self, pts):
        """
        Returns a mask that has the shape of `pts.shape[:-1]`, where False elements represent points that have aLready
        been measured in the past and should not be measured again.

        :param pts: Tensor. Tensor of suggested points with shape `[n, q, d]`.
        :return: Tensor. A mask indicating non-duplicating points.
        """
        if len(self.measured_points) == 0:
            return torch.ones(pts.shape[:-1]).bool()
        diff_mat = torch.abs(pts[..., None] - self.measured_points)
        diff_mat = (diff_mat < self.duplicate_distance_threshold).int()
        # The first sum is over feaature dim and the second is over measured_points.
        diff_mat = diff_mat.sum(-1).sum(-1).bool()
        mask = ~diff_mat
        if torch.count_nonzero(mask) == 0:
            warnings.warn('All suggested points have been measured in the past! ({})'.format(pts))
            mask = torch.ones(pts.shape[:-1]).bool()
        return mask


class BoTorchOptimizer(Optimizer):
    """
    Wrapper of BoTorch optimization functions.
    """
    default_params = {
        optimize_acqf: {
            "num_restarts": 5,
            "raw_samples": 10
        },
        optimize_acqf_discrete: {}
    }

    required_params = {
        optimize_acqf: ['bounds', 'q', 'num_restarts'],
        optimize_acqf_discrete: ['q']
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
        if self.optim_func in self.default_params.keys():
            for arg in self.default_params[self.optim_func].keys():
                if arg not in arg_dict.keys():
                    arg_dict[arg] = self.default_params[self.optim_func][arg]

        if self.optim_func in self.required_params.keys():
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
        Maximize the acquisition function. Returns the points of optima, and the corresponding acquisition
        function value. The returned points have shape `[num_candidates = q, d]`. Regardless of `q`, the returned
        acquisition value is always a scalar that gives the highest acquisition value among the `q` points.

        The specified BoTorch optimization function is used to get the optima. By default, num_restarts is set to
        a number greater than 1, so that the function returns a Tensor of [num_restarts, q, d]. Then, points
        that have been measured in the past are identified, and the q-batches containing such points are excluded.
        The suggested point(s) are then the q-batch that has the largest acquisition value among what remain.
        The stored list of measured points is updated with the selected point(s).

        :param acquisition_function: AcquisitionFunction. The acquisition function to optimize.
        :return: Tensor, Tensor[float]. The locations and value of the optima.
        """
        arg_dict = self.get_argument_dict()
        # If using optimize_acqf, returns optimal points in [num_restarts, q = num_candidates, d] and their
        # acquisition values in [num_restarts] (there is no q dimension).
        # If using optimize_discrete, the shape of returned points will be [q, d].
        pts, acq_vals = self.optim_func(acquisition_function, return_best_only=False, **arg_dict)
        if pts.ndim == 2:
            pts = pts[None, ...]
        if acq_vals.ndim == 0:
            acq_vals = torch.tensor([[acq_vals]])
        elif acq_vals.ndim == 1:
            acq_vals = acq_vals[None, ...]
        # nonduplicating_mask.shape = [num_restarts, q].
        nonduplicating_mask = self.find_duplicate_point_mask(pts)

        # Find the q-batch that has the highest acquisition value after exlcuding those containing already measured
        # points.
        q_selection_mask = nonduplicating_mask.int().sum(-1).bool()
        acq_vals[~q_selection_mask] = -torch.inf
        restart_ind = torch.argmax(acq_vals)
        pts = pts[restart_ind]
        acq_vals = acq_vals[restart_ind]

        self.update_sampled_points(pts)
        return pts, acq_vals

    def get_required_params(self):
        return self.required_params[self.optim_func]
