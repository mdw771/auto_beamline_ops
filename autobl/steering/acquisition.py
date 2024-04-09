from typing import Optional

import botorch
import torch
from botorch.models.model import Model
from botorch.acquisition import *
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from autobl.util import *


class FittingResiduePosteriorStandardDeviation(PosteriorStandardDeviation):
    r"""Posterior standard deviation enhanced by fitting residue.
    The current posterior mean is linearly fit using a series of reference spectra, and
    the fitting residue is combined with the posterior mean to give the acquisiion function's
    value.
    """
    def __init__(
            self, model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            reference_spectra_x: Tensor = None,
            reference_spectra_y: Tensor = None,
            phi: float = 0.1
    ) -> None:
        """
        The constructor.

        :param reference_spectra_x: Tensor. A Tensor with shape [m,] containing the coordinates of the
                                    n refernce spectra.
        :param reference_spectra_y: Tensor. A Tensor with shape [n, m] containing the values of the
                                    n refernce spectra.
        :param phi: Optional[float]. Weight of the residue term.
        """
        super().__init__(model, posterior_transform, maximize)
        self.reference_spectra_x = reference_spectra_x
        self.reference_spectra_y = reference_spectra_y
        self.phi = phi

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        import matplotlib.pyplot as plt

        mu, _ = self._mean_and_sigma(self.reference_spectra_x)
        amat = self.reference_spectra_y.T
        bvec = mu.reshape(-1, 1)
        xvec = torch.matmul(torch.linalg.pinv(amat), bvec)
        y_fit = torch.matmul(amat, xvec).view(-1)
        res = (y_fit - mu) ** 2

        # plt.plot(self.reference_spectra_y[0].squeeze().cpu(), linestyle='--')
        # plt.plot(self.reference_spectra_y[1].squeeze().cpu(), linestyle='--')
        # plt.show()

        _, sigma = self._mean_and_sigma(x)
        r = interp1d_tensor(self.reference_spectra_x.squeeze(),
                            res,
                            x.squeeze())
        a = sigma + self.phi * r
        # if len(x) > 100:
        #     plt.plot(y_fit.squeeze().detach().cpu())
        #     plt.plot(mu.squeeze().detach().cpu())
        #     plt.plot(sigma.squeeze().detach().cpu())
        #     plt.plot(r.squeeze().detach().cpu() * self.phi)
        #     plt.plot(a.squeeze().detach().cpu())
        #     plt.show()
        return a


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
