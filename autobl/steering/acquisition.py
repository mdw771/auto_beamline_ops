from typing import Optional, Callable

import botorch
import matplotlib.pyplot as plt
import torch
from botorch.models.model import Model
from botorch.acquisition import *
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from autobl.util import *


class PosteriorStandardDeviationDerivedAcquisition(PosteriorStandardDeviation):
    def __init__(
            self, model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            add_posterior_stddev: bool = True,
            debug=False
    ) -> None:
        """
        The constructor.

        :param beta: float. Decay factor of the weights of add-on terms in the acquisition function.
        :param add_posterior_stddev: bool. If False, the posterior standard deviation will not be added
            to the returned value, and the acquisition function will be solely
            contributed by fitting residue.
        """
        super().__init__(model, posterior_transform, maximize)
        self.add_posterior_stddev = add_posterior_stddev
        self.debug = debug
        self.mask_func = None
        self.phi = 0
        self.beta = beta

    def set_mask_func(self, f: Callable):
        self.mask_func = f

    def apply_mask_func(self, x, a):
        if self.mask_func is not None:
            m = self.mask_func(x).squeeze(-2).squeeze(-1)
            a = a * m
        return a

    def update_hyperparams_following_schedule(self, i_iter):
        self.phi = self.phi * self.beta ** i_iter


class FittingResiduePosteriorStandardDeviation(PosteriorStandardDeviationDerivedAcquisition):
    r"""Posterior standard deviation enhanced by fitting residue.
    The current posterior mean is linearly fit using a series of reference spectra, and
    the fitting residue is combined with the posterior mean to give the acquisiion function's
    value.
    """
    def __init__(
            self, model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            reference_spectra_x: Tensor = None,
            reference_spectra_y: Tensor = None,
            phi: float = 0.1,
            add_posterior_stddev: bool = True,
            debug: bool = False
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
        super().__init__(model, posterior_transform, maximize, beta, add_posterior_stddev, debug)
        self.reference_spectra_x = reference_spectra_x
        self.reference_spectra_y = reference_spectra_y
        self.phi = phi

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
        a = self.apply_mask_func(x, a)

        if self.debug:
            self._plot_acquisition_values(locals())
        return a

    def _plot_acquisition_values(self, func_locals):
        mu = func_locals['mu']
        y_fit = func_locals['y_fit']
        res = func_locals['res']
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(mu.squeeze()))
        ax.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(y_fit.squeeze()))
        ax2 = ax.twinx()
        ax2.plot(to_numpy(self.reference_spectra_x.squeeze()), to_numpy(res.squeeze()), color='red')
        plt.show()


class GradientAwarePosteriorStandardDeviation(PosteriorStandardDeviationDerivedAcquisition):
    r"""Gradient-aware posterior standard deviation.
    The acquisition function is given as sigma + phi * grad, where grad
    is the norm of the gradient of the posterior mean along given axes, and
    phi is the weight.
    """
    def __init__(
            self, model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            gradient_dims: Optional[list[int, ...]] = None,
            phi: float = 0.1,
            phi2: float = 0.001,
            method: str = 'analytical',
            order: int = 1,
            finite_difference_interval: float = 1e-2,
            add_posterior_stddev: bool = True,
            debug: bool = False
    ) -> None:
        """
        The constructor.

        :param gradient_dims: Optional[list[int, ...]]. The dimensions along which the magnitude of gradient should
            be computed. If None, it will be assumed to be the last dimension.
        :param phi: float. The weight of the gradient term.
        :param phi2: float. The weight of the second or higher-order derivative term. Disregarded if `order == 1`.
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
        super().__init__(model, posterior_transform, maximize, beta, add_posterior_stddev, debug)
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

    def update_hyperparams_following_schedule(self, i_iter):
        self.phi = self.phi * self.beta ** i_iter
        self.phi2 = self.phi2 * self.beta ** i_iter

    @t_batch_mode_transform()
    def forward(self, x: Tensor, mu_x=None, sigma_x=None) -> Tensor:
        if mu_x is None or sigma_x is None:
            mu, sigma = self._mean_and_sigma(x)
        else:
            mu, sigma = mu_x, sigma_x
        gradients_all_orders = [0.0] * self.order
        if self.method == 'analytical':
            g = self.calculate_gradients_analytical(x)
            gradients_all_orders[0] = torch.linalg.norm(g, dim=-1)
        elif self.method == 'numerical':
            for ord in range(1, self.order + 1):
                g = self.calculate_gradients_numerical(x, order=ord)
                g = torch.linalg.norm(g, dim=-1)
                gradients_all_orders[ord - 1] = g
        else:
            raise ValueError
        if not self.add_posterior_stddev:
            sigma = 0
        a = sigma + self.phi * gradients_all_orders[0]
        if len(gradients_all_orders) > 1:
            for gg in gradients_all_orders[1:]:
                a = a + self.phi2 * gg
        a = self.apply_mask_func(x, a)
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
        def differentiate(x, h, f, order=1):
            if order == 1:
                return (f(x + h) - f(x - h)) / (2 * h)
            else:
                d = (differentiate(x + h, h, f, order=order - 1) - differentiate(x - h, h, f, order=order - 1))
                d = d / (2 * h)
                return d
        f = lambda x: self.model.posterior(x).mean
        g = []
        gradient_dims = self.gradient_dims
        if gradient_dims is None:
            gradient_dims = list(range(x.shape[-1]))
        for grad_dim in gradient_dims:
            h = torch.zeros(x.shape[-1])
            h[grad_dim] = self.finite_difference_interval
            gi = differentiate(x, h, f, order=order)
            g.append(gi)
        g = torch.cat(g, dim=-1)
        return g.squeeze(1)


class ComprehensiveAigmentedAcquisitionFunction(PosteriorStandardDeviationDerivedAcquisition):
    r"""Acquisition function that combines gradient and reference spectrum augmentations."""
    def __init__(
            self,
            model: Model,
            posterior_transform: Optional[PosteriorTransform] = None,
            maximize: bool = True,
            beta: float = 0.99,
            gradient_dims: Optional[list[int, ...]] = None,
            gradient_order: int = 1,
            differentiation_method: str = 'analytical',
            reference_spectra_x: Tensor = None,
            reference_spectra_y: Tensor = None,
            phi_g: float = 0.1,
            phi_g2: float = 0.001,
            phi_r: float = 100,
            addon_term_lower_bound: float = 1e-2,
            add_posterior_stddev: bool = True,
            debug: bool = False
    ) -> None:
        """
        The constructor.

        :param phi_g: float.
        :param phi_g2: float.
        :param phi_r: float.
        :param addon_term_lower_bound: float. The lower bound of add-on terms like gradient and fitting residue. The
            weighted sum of these terms is clipped to this value before multiplied with the posterior standard
            deviation. Choose this value carefully: a too large value turns the add-on terms to a constant and the
            acquisition function will behave like a simple posterior standard deviation, while a too small value
            prevents the algorithm from exploring regions with high uncertainty yet low add-on term values.
        """
        super().__init__(model, posterior_transform, maximize, beta, add_posterior_stddev, debug)
        self.acqf_g = GradientAwarePosteriorStandardDeviation(
            model,
            posterior_transform=posterior_transform,
            maximize=maximize,
            beta=beta,
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
            beta=beta,
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
        self.addon_term_lower_bound = addon_term_lower_bound
        self.debug = debug

    def update_hyperparams_following_schedule(self, i_iter):
        self.acqf_r.update_hyperparams_following_schedule(i_iter)
        self.acqf_g.update_hyperparams_following_schedule(i_iter)

    @t_batch_mode_transform()
    def forward(self, x: Tensor) -> Tensor:
        mu, sigma = self._mean_and_sigma(x)
        if self.add_posterior_stddev:
            a = sigma - sigma.min()
        else:
            a = 1.0
        if self.phi_g > 0:
            a_g = self.acqf_g(x, mu_x=mu, sigma_x=sigma)
        else:
            a_g = torch.tensor(0.0, device=x.device)
        if self.phi_r > 0:
            a_r = self.acqf_r(x, mu_x=mu, sigma_x=sigma)
        else:
            a_r = torch.tensor(0.0, device=x.device)
        a = a * torch.clip(a_g + a_r, self.addon_term_lower_bound, None)
        a = self.apply_mask_func(x, a)
        if self.debug:
            self._plot_acquisition_values(locals())
        return a

    def _plot_acquisition_values(self, func_locals):
        mu = func_locals['mu']
        sigma = func_locals['sigma']
        a_g = func_locals['a_g']
        a_r = func_locals['a_r']
        a = func_locals['a']
        x = func_locals['x']
        x_squeezed = to_numpy(x.squeeze())
        if len(x) > 100:
            fig, ax = plt.subplots(5, 1)
            ax[0].plot(x_squeezed, to_numpy(mu.squeeze()))
            ax[0].fill_between(
                x_squeezed, to_numpy((mu - sigma).squeeze()), to_numpy((mu + sigma).squeeze()), alpha=0.5)
            ax[1].plot(x_squeezed, to_numpy(sigma.squeeze()))
            if self.phi_g > 0:
                ax[2].plot(x_squeezed, to_numpy(a_g.squeeze()))
            if self.phi_r > 0:
                ax[2].plot(x_squeezed, to_numpy(a_r.squeeze()))
            if self.phi_g > 0 or self.phi_r:
                ax[3].plot(x_squeezed, to_numpy((a_g + a_r).squeeze()))
            ax[4].plot(x_squeezed, to_numpy(a.squeeze()))
            if self.mask_func is not None:
                fig, ax = plt.subplots(1, 1)
                ax.plot(x_squeezed, to_numpy(self.mask_func(x.squeeze())))
            plt.show()
