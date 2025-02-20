"""Data/metadata containers for the people per structure modeling pipeline."""

from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple, TypeAlias

import shapely
import torch
from pydantic import BaseModel


class TileID(NamedTuple):
    """A unique identifier for a tile at a specific time point."""

    block_key: str
    tile_key: str
    tile_poly: shapely.Polygon
    time_point: str


class ModelSplit(list[TileID]):
    """A subset of the model tiles."""

    @property
    def tile_keys(self) -> list[str]:
        """Get all the tile keys in the split."""
        return [tile.tile_key for tile in self]

    def __repr__(self) -> str:
        return f"ModelSplit({len(self)} tiles)"


class ModelPartition(NamedTuple):
    """A partition of the model data into training, validation, and test splits."""

    train: ModelSplit
    validate: ModelSplit
    test: ModelSplit

    @property
    def all_tile_keys(self) -> list[TileID]:
        """Get all the tile keys in the split."""
        return self.train + self.validate + self.test


class Metric(StrEnum):
    """An enumeration of the different metrics."""

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    NRMSD = "nrmsd"
    MAPE = "mape"
    SMAPE = "smape"
    R2 = "r2"


MetricFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ScoreFunction: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


class ModelTarget(StrEnum):
    """An enumeration of the different training and scoring targets."""

    PIXEL_OCCUPANCY_RATE = "pixel_occupancy_rate"
    PIXEL_LOG_OCCUPANCY_RATE = "pixel_log_occupancy_rate"
    PIXEL_POPULATION = "pixel_population"
    PIXEL_POPULATION_DENSITY = "pixel_population_density"
    PIXEL_LOG_POPULATION_DENSITY = "log_population_density"
    ADMIN_OCCUPANCY_RATE = "admin_occupancy_rate"
    ADMIN_LOG_OCCUPANCY_RATE = "admin_log_occupancy_rate"
    ADMIN_POPULATION = "admin_population"
    ADMIN_POPULATION_DENSITY = "admin_population_density"
    ADMIN_LOG_POPULATION_DENSITY = "admin_log_population_density"


class ModelSpecification(BaseModel):
    model_version: str
    model_root: str
    output_root: str
    denominator: str
    resolution: str
    features: list[str]

    split: int = 0  # Which test/train/validate split to use
    # What the model predicts from the covariates in training
    training_target: ModelTarget = ModelTarget.ADMIN_LOG_OCCUPANCY_RATE
    # What the loss function evaluates (may be a transformation of the training target)
    loss_target: ModelTarget = ModelTarget.ADMIN_LOG_OCCUPANCY_RATE
    # The metric to use for the loss function
    loss_metric: Metric = Metric.MAE

    # TODO: The selection of training/loss constricts the options here. Need a
    #   more restrictive enumeration of the possible targets conditional on the
    #   training/loss targets
    # What the model predicts in testing (should be same measure, but may be
    # at a different level; e.g. may train at admin and predict at pixel)
    test_target: ModelTarget = ModelTarget.ADMIN_LOG_OCCUPANCY_RATE
    # What the evaluation function evaluates (may be a transformation of the
    # test target)
    evaluation_target: ModelTarget = ModelTarget.ADMIN_LOG_OCCUPANCY_RATE
    # The metric to use for the evaluation function
    evaluation_metric: Metric = Metric.MAE

    # Model parameters
    # The random seed for the model
    random_seed: int = 112358
    # The model type to run.  Currently only linear regression is supported
    mtype: str = "linear"
    # The optimizer and optimizer parameters
    optimizer: str = "adam"
    learning_rate: float = 0.001

    # Data loading parameters
    num_cores: int = 8
