import dataclasses
from collections.abc import Sequence, Callable
from typing import Type, Optional, Any
import json

import botorch
import gpytorch

from autobl.steering.optimization import Optimizer, ContinuousOptimizer


@dataclasses.dataclass
class Config:

    random_seed: Any = 123
    """Random seed."""

    def dictionarize(self):
        d = {}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], Config):
                d[key] = self.__dict__[key].dictionarize()
            else:
                d[key] = self.__dict__[key]
        return d

    def to_json(self, fname):
        json.dump(self.dictionarize(), open(fname, 'w'))


@dataclasses.dataclass
class ExperimentGuideConfig(Config):

    dim_measurement_space: int = None
    """Number of dimensions of the space where measurements are performed (i.e., the feature space)."""

    lower_bounds: Sequence[float, ...] = None
    """Lower bound of sampling points to be suggested."""

    upper_bounds: Sequence[float, ...] = None
    """Upper bound of sampling points to be suggested."""

    num_candidates: int = 1
    """
    Number of sampling points to suggest. If an analytical acquisition function built in BoTorch is used, this 
    must be 1; to get multiple candidates, either use Monte-Carlo-based acquisition functions (e.g., 
    those whose names start with "q" like `qUpperConfidenceBound`), or define a custom acquisition function.
    """

    def __post_init__(self):
        if self.dim_measurement_space is None:
            raise TypeError("Missing required argument: 'dim_measurement_space'")


@dataclasses.dataclass
class GPExperimentGuideConfig(ExperimentGuideConfig):

    model_class: Type[botorch.models.model.Model] = botorch.models.SingleTaskGP
    """Class handle of the model. Should be a subclass of botorch.Model."""

    model_params: dict = dataclasses.field(default_factory=dict)
    """
    Parameters of the model class. 
    
    The covariance kernel and its parameters also go here and should be given through 
    "covariance_module" as a `gpytorch.kernels.Kernel` object. One may also specify a prior for
    the lengthscale when instantiating the kernel. The distribution parameters should be in the space normalized
    between 0 and 1. For example:
    ```
    model_params={'covar_module': gpytorch.kernels.MaternKernel(
        nu=2.5, 
        lengthscale_prior=gpytorch.priors.NormalPrior(0.08, 0.05))}
    ```
    Instead of using a prior, one may also use the `override_kernel_lengthscale` parameter to specify a fixed
    value for the lengthscale. 
    """

    acquisition_function_class: Type[botorch.acquisition.acquisition.AcquisitionFunction] = (
        botorch.acquisition.UpperConfidenceBound)
    """Class handle of the acquisition function. Should be an instance of botorch.AcquisitionFunction."""

    acquisition_function_params: dict = dataclasses.field(default_factory=dict)
    """Parameters of the acquisition function."""

    optimizer_class: Type[Optimizer] = ContinuousOptimizer
    """
    The function handle of the optimization function. The function should have the acquisition function,
    q (number of candidates) and other parameters as arguments.
    """

    optimizer_params: dict = dataclasses.field(default_factory=dict)
    """
    Parameters of the optimizer constructor, not including `bounds`, `num_candidates` (these arguments are filled
    in in the ExperimentGuide class based on other config settings).
    """

    override_kernel_lengthscale: Optional[float] = None
    """
    If given, the lengthscale parameter of the kernel will be overriden with this number, but its prior distribution
    will not be changed. This value should be given in the original scale of the data, i.e., without any normalization.
    """

    noise_variance: Optional[float] = None
    """Noise variance of the observations."""

    beta: float = 0.99
    """Decay factor of the weights of add-on terms in the acquisition function."""

    def __post_init__(self):
        super().__post_init__()
        if 'input_transform' in self.model_params.keys():
            raise ValueError("I see you specified input_transform in model_params. Don't do it! Data are normalized/"
                             "standaridized in GPExperimentGuide automatically.")
        if 'outcome_transform' in self.model_params.keys():
            raise ValueError("I see you specified outcome_transform in model_params. Don't do it! Data are normalized/"
                             "standaridized in GPExperimentGuide automatically.")


@dataclasses.dataclass
class XANESExperimentGuideConfig(GPExperimentGuideConfig):

    n_updates_create_acqf_mask_func: Optional[int] = None
    """
    If provided, the guide builds a sigmoid function that attenuates acquisition function values in pre-edge regions
    after this number of model updates.
    """

    acqf_mask_floor_value: float = 0.1
    """Floor value of the sigmoid function used as acquisition function mask."""

    acqf_mask_post_edge_gain: float = 5.0
    """Post edge gain in acquisition mask function."""

    acqf_mask_post_edge_offset: float = 1.0
    """Location of post edge gain in acquisition mask function as a multiple of edge width."""

    acqf_mask_post_edge_width: float = 0.5
    """Width of post edge gain in acquisition mask function as a multiple of edge width."""
