from typing import Optional

import botorch
import torch
from botorch.models.model import Model
from botorch.acquisition import *
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class GradientAwarePosteriorStandardDeviation(PosteriorStandardDeviation):
    r"""Gradient-aware posterior standard deviation.
    The acquisition function is given as sigma + phi * grad, where grad
    is the norm of the gradient of the posterior mean along given axes, and
    phi is the weight.
    """
    def __init__(
            self, model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            gradient_dims: Optional[list[int, ...]] = None,
            phi: float = 0.1
    ) -> None:
        super().__init__(model, posterior_transform, maximize)
        self.gradient_dims = gradient_dims
        self.phi = phi

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        mu, sigma = self._mean_and_sigma(x)
        g = self.calculate_gradients(x)
        g = torch.linalg.norm(g, dim=-1)
        a = sigma + self.phi * g
        return a

    def calculate_gradients(self, x: Tensor):
        f_posterior_mean = lambda x: self.model.posterior(x).mean.squeeze()
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f_posterior_mean, x)
            g = jac[torch.tensor(range(jac.shape[0])), torch.tensor(range(jac.shape[0]))]
        if self.gradient_dims is not None:
            g = torch.index_select(g, -1, self.gradient_dims)
        return g.squeeze(1)
