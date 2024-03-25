import botorch
import gpytorch
import numpy as np
import torch
import matplotlib.pyplot as plt

from autobl.steering.configs import *

class ExperimentGuide:

    def __init__(self, config: ExperimentGuideConfig, *args, **kwargs):
        self.config = config

    def build(self, *args, **kwargs):
        pass

    def suggest(self):
        pass

    def update(self):
        pass

    def get_bounds(self):
        lb = self.config.lower_bounds
        if lb is None:
            lb = torch.tensor([-np.inf] * len(self.config.measurement_space_dims))
        ub = self.config.upper_bounds
        if ub is None:
            ub = torch.tensor([np.inf] * len(self.config.measurement_space_dims))
        if not isinstance(lb, torch.Tensor):
            lb = torch.from_numpy(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.from_numpy(ub)
        return torch.stack([lb, ub])


class GPExperimentGuide(ExperimentGuide):

    def __init__(self, config: GPExperimentGuideConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = None
        self.fitting_func = None
        self.acquisition_function = None
        self.data_x = torch.tensor([])
        self.data_y = torch.tensor([])

    def build(self, initial_data):
        """
        Build model, fit hyperparameters, and initialize other variables.

        :param initial_data: Optional[tuple[np.ndarray, np.ndarray]]. Features and observation data for training
               the GP model to find hyperparameters (e.g., kernel paraemters).
        """
        self.build_model(train_data=initial_data)
        self.build_acquisition_function()

    def record_data(self, x, y):
        self.data_x = torch.concatenate([self.data_x, x])
        self.data_y = torch.concatenate([self.data_y, y])

    def build_model(self, train_data):
        x_train, y_train = train_data
        self.model = self.config.model_class(x_train, y_train, **self.config.model_params)
        self.fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_mll(self.fitting_func)
        self.record_data(x_train, y_train)

    def build_acquisition_function(self):
        self.acquisition_function = self.config.acquisition_function_class(self.model,
                                                                           **self.config.acquisition_function_params)

    def suggest(self):
        candidate, acq_val = botorch.optim.optimize_acqf(
            self.acquisition_function,
            bounds=self.get_bounds(),
            num_restarts=5,
            q=self.config.num_candidates,
            raw_samples=10
        )
        return candidate

    def plot_posterior(self, x):
        """
        Plot the posterior mean and standard deviation of the GP model. Only works with 1-dimension feature space.

        :param x: torch.Tensor[float, ...]. The points to plot.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        posterior = self.model.posterior(x)
        mu = posterior.mean.reshape(-1).cpu().detach().numpy()
        sigma2 = posterior.variance.reshape(-1).cpu().detach().numpy()

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, mu)
        ax.fill_between(x, mu - np.sqrt(sigma2), mu + np.sqrt(sigma2), alpha=0.5)
        ax.scatter(self.data_x.reshape(-1), self.data_y.reshape(-1))
        plt.show()
