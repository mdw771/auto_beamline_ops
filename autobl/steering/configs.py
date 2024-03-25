import dataclasses
from collections.abc import Sequence
from typing import Type, Optional, Any

import botorch
import gpytorch


@dataclasses.dataclass
class Config:

    random_seed: Any = 123
    """Random seed."""


@dataclasses.dataclass
class ExperimentGuideConfig(Config):

    measurement_space_dims: Sequence[int, ...] = None
    """Dimensions of the space where measurements are performed."""

    lower_bounds: Sequence[float, ...] = None
    """Lower bound of sampling points to be suggested."""

    upper_bounds: Sequence[float, ...] = None
    """Upper bound of sampling points to be suggested."""

    num_candidates: int = 1
    """Number of sampling points to suggest."""

    def __post_init__(self):
        if self.measurement_space_dims is None:
            raise TypeError("Missing required argument: 'measurement_space_dims'")


@dataclasses.dataclass
class GPExperimentGuideConfig(ExperimentGuideConfig):

    model_class: Type[botorch.models.model.Model] = botorch.models.SingleTaskGP
    """Class handle of the model. Should be a subclass of botorch.Model."""

    model_params: dict = dataclasses.field(default_factory=dict)
    """Parameters of the model class."""

    acquisition_function_class: Type[botorch.acquisition.acquisition.AcquisitionFunction] = (
        botorch.acquisition.UpperConfidenceBound)
    """Class handle of the acquisition function. Should be an instance of botorch.AcquisitionFunction."""

    acquisition_function_params: dict = dataclasses.field(default_factory=dict)
    """Parameters of the acquisition function."""
