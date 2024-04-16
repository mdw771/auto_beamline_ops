"""
Ref: https://github.com/saugatkandel/AI-ML_Control_System/blob/66ed73afa80d2746bae8126d0cbe3c0ea570f141/work_directory/34-ID/jupyter/botorch_test/turbo_1.ipynb#L34
"""
import logging

import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
import gpytorch
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt

from autobl.steering.configs import *
from autobl.util import *


class ExperimentGuide:

    def __init__(self, config: ExperimentGuideConfig, *args, **kwargs):
        self.config = config

    def build(self, *args, **kwargs):
        pass

    def suggest(self):
        pass

    def update(self, *args, **kwargs):
        pass


class GPExperimentGuide(ExperimentGuide):

    def __init__(self, config: GPExperimentGuideConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = None
        self.fitting_func = None
        self.acquisition_function = None
        self.optimizer = None
        self.data_x = torch.tensor([])
        self.data_y = torch.tensor([])
        self.input_transform = None
        self.outcome_transform = None
        self.n_suggest_calls = 0
        self.n_update_calls = 0

    def build(self, x_train=None, y_train=None):
        """
        Build model, fit hyperparameters, and initialize other variables.

        :param x_train: Optional[Tensor]. Features of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        :param y_train: Optional[Tensor]. Observations of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        """
        self.build_counters()
        self.build_transform()
        self.build_model(x_train, y_train)
        self.build_acquisition_function()
        self.build_optimizer()

    def build_counters(self):
        self.n_suggest_calls = 0
        self.n_update_calls = 0

    def build_transform(self):
        self.input_transform = Normalize(d=self.config.dim_measurement_space)
        self.outcome_transform = Standardize(m=self.config.dim_measurement_space)

    def record_data(self, x, y):
        self.data_x = torch.concatenate([self.data_x, x])
        self.data_y = torch.concatenate([self.data_y, y])

    def build_model(self, x_train, y_train):
        x_train, y_train = self.transform_data(x_train, y_train, train=True)
        self.train_model(x_train, y_train)
        self.record_data(x_train, y_train)

    def build_acquisition_function(self):
        if issubclass(self.config.acquisition_function_class, botorch.acquisition.AnalyticAcquisitionFunction):
            assert self.config.num_candidates == 1, ('Since an analytical acquisition function is used, '
                                                     'num_candidates must be 1.')
        self.acquisition_function = self.config.acquisition_function_class(
            self.model,
            **self.config.acquisition_function_params,
            posterior_transform=botorch.acquisition.objective.UnstandardizePosteriorTransform(
                Y_mean=self.outcome_transform.means[0], Y_std=self.outcome_transform.stdvs[0])
        )
        if hasattr(self.acquisition_function, 'update_hyperparams_following_schedule'):
            self.acquisition_function.update_hyperparams_following_schedule(self.n_update_calls)

    def build_optimizer(self):
        self.optimizer = self.config.optimizer_class(
            bounds=self.get_bounds(),
            num_candidates=self.config.num_candidates,
            **self.config.optimizer_params
        )

    def get_bounds(self):
        lb = self.config.lower_bounds
        if lb is None:
            lb = torch.tensor([-np.inf] * self.config.dim_measurement_space)
        ub = self.config.upper_bounds
        if ub is None:
            ub = torch.tensor([np.inf] * self.config.dim_measurement_space)
        if not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb)
        lb, _ = self.transform_data(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub)
        ub, _ = self.transform_data(ub)
        return torch.stack([lb, ub])

    def scale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data by 1 / span_of_normalizer_bounds.

        :param x: Any. The input data.
        :param dim: int. Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
                    If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
                    the scaling factors are calculated separately for each dimension and the `dim` argument is
                    disregarded.
        :return:
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x / (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = self.input_transform.bounds[1][dim] - self.input_transform.bounds[0][dim]
            return x / s

    def unscale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data by span_of_normalizer_bounds.

        :param x: Any. The input data.
        :param dim: int. Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
                    If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
                    the scaling factors are calculated separately for each dimension and the `dim` argument is
                    disregarded.
        :return:
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x * (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = self.input_transform.bounds[1][dim] - self.input_transform.bounds[0][dim]
            return x * s

    def transform_data(self, x=None, y=None, train=False):
        if x is not None:
            x = x.double()
        if y is not None:
            y = y.double()
        if x is not None and self.input_transform is not None:
            do_squeeze = False
            if x.ndim == 1:
                x = x[:, None]
                do_squeeze = True
            if train:
                self.input_transform.train()
            else:
                self.input_transform.eval()
            x = self.input_transform(x)
            if do_squeeze:
                x = x[:, 0]
        if y is not None and self.outcome_transform is not None:
            if train:
                self.outcome_transform.train()
            else:
                self.outcome_transform.eval()
            y, _ = self.outcome_transform(y)
        return x, y

    def untransform_data(self, x=None, y=None):
        if x is not None and self.input_transform is not None:
            self.input_transform.training = False
            x = self.input_transform.untransform(x)
        if y is not None and self.outcome_transform is not None:
            self.outcome_transform.training = False
            y, _ = self.outcome_transform.untransform(y)
        return x, y

    def untransform_posterior(self, posterior):
        posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def suggest(self):
        candidate, acq_val = self.optimizer.maximize(self.acquisition_function)
        candidate, _ = self.untransform_data(candidate)
        self.n_suggest_calls += 1
        return candidate

    def update(self, x_data, y_data):
        """
        Update the model using newly measured data.

        :param x_data: Tensor. Features of new data.
        :param y_data: Tensor. Observations of new data.
        """
        x_data, y_data = self.transform_data(x_data, y_data, train=False)
        self.record_data(x_data, y_data)
        additional_params = {}
        if self.config.noise_variance is not None:
            additional_params['noise'] = torch.full_like(y_data, self.config.noise_variance)
        self.model = self.model.condition_on_observations(x_data, y_data, **additional_params)
        # condition_on_observations does not make in-place changes to the model object but creates a new object, so
        # we need to reset the model object in the acquisition function.
        self.build_acquisition_function()
        self.n_update_calls += 1

    def train_model(self, x_data, y_data):
        # Create model and compute covariance matrix.
        assert not ('train_Yvar' in self.config.model_params.keys() and self.config.noise_variance is not None)
        additional_params = {}
        if self.config.noise_variance is not None:
            additional_params['train_Yvar'] = torch.full_like(y_data, self.config.noise_variance)
        self.model = self.config.model_class(x_data, y_data, **self.config.model_params, **additional_params)

        # Fit hyperparameters.
        logging.info('Kernel lengthscale before optimization (normalized & standardized): {}'.format(
            to_numpy(self.model.covar_module.lengthscale))
        )
        self.fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_mll(self.fitting_func)
        logging.info('Kernel lengthscale after optimization (normalized & standardized): {}'.format(
            to_numpy(self.model.covar_module.lengthscale))
        )

        # Override kernel lengthscale if applicable.
        if self.config.override_kernel_lengthscale is not None:
            self.model.covar_module.lengthscale = self.scale_by_normalizer_bounds(
                self.config.override_kernel_lengthscale)
            logging.info('Kernel lengthscale overriden to: {} ({} after normalization)'.format(
                self.config.override_kernel_lengthscale,
                self.scale_by_normalizer_bounds(self.config.override_kernel_lengthscale)))

    def get_posterior_mean_and_std(self, x, transform=True, untransform=True):
        if transform:
            x_transformed, _ = self.transform_data(x, None, train=False)
        else:
            x_transformed = x
        posterior = self.model.posterior(x_transformed)
        if untransform:
            posterior = self.untransform_posterior(posterior)
        mu = posterior.mean
        sigma = posterior.variance.clamp_min(1e-12).sqrt()
        return mu, sigma

    def plot_posterior(self, x, ax=None):
        """
        Plot the posterior mean and standard deviation of the GP model. Only works with 1-dimension feature space.

        :param x: torch.Tensor[float, ...]. The points to plot.
        """
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        if x.ndim == 1:
            x = x[:, None]
        mu, sigma = self.get_posterior_mean_and_std(x)
        mu = mu.reshape(-1).cpu().detach().numpy()
        sigma = sigma.reshape(-1).cpu().detach().numpy()
        x_transformed, _ = self.transform_data(x, None, train=False)
        acq = self.acquisition_function(x_transformed.view(-1, 1, 1)).reshape(-1).cpu().detach().numpy()

        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        external_ax = True
        if ax is None:
            external_ax = False
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        if not isinstance(ax, (list, tuple, np.ndarray)):
            ax = [ax]
        ax[0].plot(x, mu, label='Posterior mean')
        ax[0].fill_between(x, mu - sigma, mu + sigma, alpha=0.5)
        data_x, data_y = self.untransform_data(self.data_x, self.data_y)
        ax[0].scatter(to_numpy(data_x.reshape(-1)), to_numpy(data_y.reshape(-1)), label='Measured data')
        ax[0].set_title('Posterior mean and $+/-\sigma$ interval')
        if len(ax) > 1:
            ax[1].plot(x, acq)
            ax[1].set_title('Acquisition function')
        if external_ax:
            return ax
        else:
            plt.show()


class XANESExperimentGuide(GPExperimentGuide):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acqf_mask_func = None

    def update(self, x_data, y_data):
        super().update(x_data, y_data)
        if self.config.n_updates_create_acqf_mask_func is not None and \
                self.n_update_calls == self.config.n_updates_create_acqf_mask_func:
            logging.info('Building acquisition function mask with floor value {}.'.format(self.config.acqf_mask_floor_value))
            self.build_acqf_mask_function(floor_value=self.config.acqf_mask_floor_value)

    def build_acquisition_function(self):
        super().build_acquisition_function()
        if hasattr(self.acquisition_function, 'set_mask_func'):
            self.acquisition_function.set_mask_func(self.acqf_mask_func)

    def build_acqf_mask_function(self, floor_value=0.1):
        """
        Create and set a mask function to the acquisition function, such that it lowers the values in pre-edge
        regions.
        """
        if not hasattr(self.acquisition_function, 'set_mask_func'):
            logging.warning('Acquisition function does not have attribute "set_mask_func".')
            self.acqf_mask_func = None
            return
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        mu, _ = self.get_posterior_mean_and_std(x, transform=False)
        mu = mu.squeeze()
        # convert 3 eV to pixel
        gaussian_grad_sigma = 3.0 / (self.input_transform.bounds[1][0] - self.input_transform.bounds[0][0]) * len(x)
        gaussian_grad_sigma = float(gaussian_grad_sigma)
        mu_grad = scipy.ndimage.gaussian_filter(to_numpy(mu), sigma=gaussian_grad_sigma, order=1)

        # convert 3 eV to pixel
        min_peak_width = float(3.0 / (self.input_transform.bounds[1][0] - self.input_transform.bounds[0][0]) * len(x))
        peak_inds, peak_properties = scipy.signal.find_peaks(mu_grad, height=0.05, width=min_peak_width)

        peak_loc_normalized = float(peak_inds[0]) / len(x)
        peak_width_normalized = peak_properties['widths'][0] / len(x)

        def mask_func(x):
            m = sigmoid(x, r=20. / peak_width_normalized, d=peak_loc_normalized - 1.7 * peak_width_normalized)
            m = m + gaussian(x,
                             a=self.config.acqf_mask_post_edge_gain,
                             mu=peak_loc_normalized + self.config.acqf_mask_post_edge_offset * peak_width_normalized,
                             sigma=peak_width_normalized * self.config.acqf_mask_post_edge_width,
                             c=0.0)
            m = m * (1 - floor_value) + floor_value
            return m

        self.acqf_mask_func = mask_func
        return
