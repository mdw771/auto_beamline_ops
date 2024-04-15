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
            phi: float = 0.1,
            add_posterior_stddev: bool = True
    ) -> None:
        """
        The constructor.

        :param reference_spectra_x: Tensor. A Tensor with shape [m,] containing the coordinates of the
                                    n refernce spectra.
        :param reference_spectra_y: Tensor. A Tensor with shape [n, m] containing the values of the
                                    n refernce spectra.
        :param phi: Optional[float]. Weight of the residue term.
        :param add_posterior_stddev: bool. If False, the posterior standard deviation will not be added
                                     to the returned value, and the acquisition function will be solely
                                     contributed by fitting residue.
        """
        super().__init__(model, posterior_transform, maximize)
        self.reference_spectra_x = reference_spectra_x
        self.reference_spectra_y = reference_spectra_y
        self.phi = phi
        self.add_posterior_stddev = add_posterior_stddev

    @t_batch_mode_transform()
    def forward(self, x: Tensor, sigma_x=None, **kwargs) -> Tensor:
        mu, _ = self._mean_and_sigma(self.reference_spectra_x)
        amat = self.reference_spectra_y.T
        bvec = mu.reshape(-1, 1)
        xvec = torch.matmul(torch.linalg.pinv(amat), bvec)
        y_fit = torch.matmul(amat, xvec).view(-1)
        res = (y_fit - mu) ** 2

        if self.add_posterior_stddev:
            if sigma_x is None:
                _, sigma = self._mean_and_sigma(x)
            else:
                sigma = sigma_x
        else:
            sigma = 0
        r = interp1d_tensor(self.reference_spectra_x.squeeze(),
                            res,
                            x.squeeze())
        a = sigma + self.phi * r
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
            phi: float = 0.1,
            phi2: float = 0.001,
            method: str = 'analytical',
            order: int = 1,
            finite_difference_interval: float = 1e-2,
            add_posterior_stddev: bool = True
    ) -> None:
        """
        The constructor.

        :param gradient_dims: Optional[list[int, ...]]. The dimensions along which the magnitude of gradient should
                              be computed. If None, it will be assumed to be the last dimension.
        :param phi: float. The weight of the gradient term.
        :param phi2: float. The weight of the second-order derivative term. Disregarded if `order == 1`.
        :param method: str. Can be "analytical" or "numerical". If "analytical", gradients are calculated using
                       automatic differentiation. Due to the limitation of BoTorch, only first-order derivative is
                       available using this method. If "numerical", gradients are calculated using finite difference
                       based on the evaluations of the posterior mean on `numerical_gradient_points` points.
        :param order: int. The order of the derivative. If `method` is `"analytical"`, it can only be 1.
        :param finite_difference_interval: float. The interval used for finite-difference differentiation. Only used
                                           when `method` is `"numerical"`.  This value is specified in the normalized
                                           scale (between 0 and 1).
        :param add_posterior_stddev: bool. If True, posterior standard deviation is added to the function value.
        """
        super().__init__(model, posterior_transform, maximize)
        self.gradient_dims = gradient_dims
        self.phi = phi
        self.phi2 = phi2
        self.method = method
        self.order = order
        self.finite_difference_interval = finite_difference_interval
        self.add_posterior_stddev = add_posterior_stddev
        if method == 'analytical' and order > 1:
            raise ValueError("When method is 'analytical', order can only be 1.")
        if method not in ['analytical', 'numerical']:
            raise ValueError("'method' can only be 'analytical' or 'numerical'.")

    @t_batch_mode_transform()
    def forward(self, x: Tensor, mu_x=None, sigma_x=None) -> Tensor:
        if mu_x is None or sigma_x is None:
            mu, sigma = self._mean_and_sigma(x)
        else:
            mu, sigma = mu_x, sigma_x
        gg = 0.0
        if self.method == 'analytical':
            g = self.calculate_gradients_analytical(x)
        elif self.method == 'numerical':
            g = self.calculate_gradients_numerical(x, order=1)
            if self.order > 1:
                gg = self.calculate_gradients_numerical(x, order=2)
                gg = torch.linalg.norm(gg, dim=-1)
        else:
            raise ValueError
        g = torch.linalg.norm(g, dim=-1)
        if not self.add_posterior_stddev:
            sigma = 0
        a = sigma + self.phi * g + self.phi2 * gg
        return a

    def calculate_gradients_analytical(self, x: Tensor):
        """
        Calculate gradients using AD.

        :param x: Tensor.
        :return: Tensor. Gradient of shape (n, d).
        """
        f_posterior_mean = lambda x: self.model.posterior(x).mean.squeeze()
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(f_posterior_mean, x)
            g = jac[torch.tensor(range(jac.shape[0])), torch.tensor(range(jac.shape[0]))]
        if self.gradient_dims is not None:
            g = torch.index_select(g, -1, self.gradient_dims)
        return g.squeeze(1)

    def calculate_gradients_numerical(self, x: Tensor, order: int = 1):
        f = lambda x: self.model.posterior(x).mean
        g = []
        gradient_dims = self.gradient_dims
        if gradient_dims is None:
            gradient_dims = list(range(x.shape[-1]))
        if order == 1:
            for grad_dim in gradient_dims:
                h = torch.zeros(x.shape[-1])
                h[grad_dim] = self.finite_difference_interval
                # x_minus = torch.clip(x - h, 0, 1)
                # x_plus = torch.clip(x + h, 0, 1)
                x_minus = x - h
                x_plus = x + h
                gi = (f(x_plus) - f(x_minus)) / (x_plus - x_minus)
                g.append(gi)
            g = torch.cat(g, dim=-1)
        elif order == 2:
            for grad_dim in gradient_dims:
                h = torch.zeros(x.shape[-1])
                h[grad_dim] = self.finite_difference_interval
                x_minus = x - h
                x_minus2 = x_minus - h
                x_plus = x + h
                x_plus2 = x_plus + h
                gi_minus = (f(x) - f(x_minus2)) / (x - x_minus2)
                gi_plus = (f(x_plus2) - f(x)) / (x_plus2 - x)
                gi = (gi_plus - gi_minus) / (x_plus - x_minus)
                g.append(gi)
            g = torch.cat(g, dim=-1)
        else:
            raise ValueError('Unsupported order of {}.'.format(order))
        return g.squeeze(1)

class ComprehensiveAigmentedAcquisitionFunction(PosteriorStandardDeviation):
    r"""Acquisition function that combines gradient and reference spectrum augmentations."""
    def __init__(
            self,
            model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            gradient_dims: Optional[list[int, ...]] = None,
            gradient_order: int = 1,
            differentiation_method: str = 'analytical',
            reference_spectra_x: Tensor = None,
            reference_spectra_y: Tensor = None,
            phi_g: float = 0.1,
            phi_g2: float = 0.001,
            phi_r: float = 100,
            add_posterior_stddev: bool = True
    ) -> None:
        super().__init__(model, posterior_transform, maximize)
        self.acqf_g = GradientAwarePosteriorStandardDeviation(
            model,
            posterior_transform=posterior_transform,
            maximize=maximize,
            gradient_dims=gradient_dims,
            method=differentiation_method,
            order=gradient_order,
            phi=phi_g,
            phi2=phi_g2,
            add_posterior_stddev=False
        )
        self.acqf_r = FittingResiduePosteriorStandardDeviation(
            model,
            posterior_transform=posterior_transform,
            maximize=maximize,
            reference_spectra_x=reference_spectra_x,
            reference_spectra_y=reference_spectra_y,
            phi=phi_r,
            add_posterior_stddev=False
        )
        self.gradient_order = gradient_order
        self.phi_r = phi_r
        self.phi_g = phi_g
        self.phi_g2 = phi_g2
        self.reference_spectra_x = reference_spectra_x
        self.reference_spectra_y = reference_spectra_y
        self.add_posterior_stddev = add_posterior_stddev

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        mu, sigma = self._mean_and_sigma(x)
        if self.add_posterior_stddev:
            a = sigma
        else:
            a = 0
        if self.phi_g > 0:
            a = a + self.acqf_g(x, mu_x=mu, sigma_x=sigma)
        if self.phi_r > 0 and self.reference_spectra_x is not None and self.reference_spectra_y is not None:
            a = a + self.acqf_r(x, mu_x=mu, sigma_x=sigma)
        return a
