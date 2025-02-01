"""Transformations between different metric/loss targets.

This module contains functions for transforming between different metric/loss
targets. For example, it provides functions for transforming between occupancy
rate and population density.

"""

import abc
from collections.abc import Callable

import torch

from rra_population_model.model.modeling.datamodel import (
    ModelTarget,
)


def get_transformation(
    training_target: ModelTarget,
    scoring_target: ModelTarget,
    pixel_building_density: torch.Tensor,
    admin_building_density: torch.Tensor,
    pixel_area_weights: torch.Tensor,
    pixel_area: float = 1600,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a transformations from a target measure to all measures of interest.

    Parameters
    ----------
    training_target
        The target measure to transform from. The training target is, generally,
        the measure directly predicted by the model.
    scoring_target
        The target measure to transform to. The scoring target is the measure
        used to evaluate the model, either as a loss function or as a general
        scoring criterion.
    pixel_building_density
        The pixel building density tensor.
    admin_building_density
        The administrative unit building density tensor.
    pixel_area_weights
        A sparse representation of the 2D-matrix that transforms pixel-level
        data to administrative unit-level data. The number of columns should be
        equal to the number of pixels in the tile and the number of rows should be
        equal to the number of administrative units.
    pixel_area
        The area of a pixel in square meters.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A transformation object that can be used to transform between different
        the target measure and all other measures of interest.

    """
    transformation_class = {
        ModelTarget.PIXEL_OCCUPANCY_RATE: PixelOccupancyRateTransformation,
        ModelTarget.PIXEL_LOG_OCCUPANCY_RATE: PixelLogOccupancyRateTransformation,
        ModelTarget.PIXEL_POPULATION: PixelPopulationTransformation,
        ModelTarget.PIXEL_POPULATION_DENSITY: PixelPopulationDensityTransformation,
        ModelTarget.PIXEL_LOG_POPULATION_DENSITY: PixelLogPopulationDensityTransformation,
        ModelTarget.ADMIN_OCCUPANCY_RATE: AdminOccupancyRateTransformation,
        ModelTarget.ADMIN_LOG_OCCUPANCY_RATE: AdminLogOccupancyRateTransformation,
        ModelTarget.ADMIN_POPULATION: AdminPopulationTransformation,
        ModelTarget.ADMIN_POPULATION_DENSITY: AdminPopulationDensityTransformation,
        ModelTarget.ADMIN_LOG_POPULATION_DENSITY: AdminLogPopulationDensityTransformation,
    }[training_target]

    transformation = transformation_class(  # type: ignore[abstract]
        pixel_building_density,
        admin_building_density,
        pixel_area_weights,
        pixel_area,
    ).get_transformation(scoring_target)

    return transformation


class Transformation(abc.ABC):
    """An abstract base class for transformations.

    Transformations operate on pixel-level inputs and can produce pixel-level
    or admin-level outputs. They are used to transform between different
    metric/loss targets.
    """

    def __init__(
        self,
        pixel_building_density: torch.Tensor,
        admin_building_density: torch.Tensor,
        pixel_area_weights: torch.Tensor,
        pixel_area: float,
    ) -> None:
        self.pixel_building_density = pixel_building_density
        self.admin_building_density = admin_building_density
        self.pixel_area_weights = pixel_area_weights
        self.pixel_area = torch.tensor(pixel_area, dtype=torch.float32)

    @abc.abstractmethod
    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level occupancy rate."""

    @abc.abstractmethod
    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level log occupancy rate."""

    @abc.abstractmethod
    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level population."""

    @abc.abstractmethod
    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level population density."""

    @abc.abstractmethod
    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level log population density."""

    @abc.abstractmethod
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level occupancy rate."""

    @abc.abstractmethod
    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level log occupancy rate."""

    @abc.abstractmethod
    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level population."""

    @abc.abstractmethod
    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level population density."""

    @abc.abstractmethod
    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level log population density."""

    @abc.abstractmethod
    def get_transformation(
        self,
        target: ModelTarget,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get a transformation to a specific target measure.

        Parameters
        ----------
        target
            The target measure to transform to.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            A transformation object that can be used to transform to the target
            measure.

        """


class PixelTransformation(Transformation, abc.ABC):
    """A transformation for pixel-level data."""

    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level occupancy rate."""
        return self.pixel_area_weights @ self.to_pixel_occupancy_rate(z)

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to admin-level log occupancy rate."""
        return self.to_admin_occupancy_rate(z).log()

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to an admin-level population count."""
        return self.pixel_area_weights @ self.to_pixel_population(z)

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to an admin population density."""
        return self.to_admin_population(z) / self.pixel_area

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to a log admin population density."""
        return self.to_admin_population_density(z).log()

    def get_transformation(
        self,
        target: ModelTarget,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        transformation = {
            ModelTarget.PIXEL_OCCUPANCY_RATE: self.to_pixel_occupancy_rate,
            ModelTarget.PIXEL_LOG_OCCUPANCY_RATE: self.to_pixel_log_occupancy_rate,
            ModelTarget.PIXEL_POPULATION: self.to_pixel_population,
            ModelTarget.PIXEL_POPULATION_DENSITY: self.to_pixel_population_density,
            ModelTarget.PIXEL_LOG_POPULATION_DENSITY: self.to_pixel_log_population_density,
            ModelTarget.ADMIN_OCCUPANCY_RATE: self.to_admin_occupancy_rate,
            ModelTarget.ADMIN_LOG_OCCUPANCY_RATE: self.to_admin_log_occupancy_rate,
            ModelTarget.ADMIN_POPULATION: self.to_admin_population,
            ModelTarget.ADMIN_POPULATION_DENSITY: self.to_admin_population_density,
            ModelTarget.ADMIN_LOG_POPULATION_DENSITY: self.to_admin_log_population_density,
        }.get(target)

        if transformation is None:
            msg = f"Invalid target: {target}"
            raise ValueError(msg)
        return transformation


class PixelOccupancyRateTransformation(PixelTransformation):
    """A transformation for occupancy rate."""

    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log()

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.pixel_building_density * self.pixel_area

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.pixel_building_density

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() + self.pixel_building_density.log()


class PixelLogOccupancyRateTransformation(PixelTransformation):
    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp()

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.pixel_building_density * self.pixel_area

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.pixel_building_density

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.pixel_building_density.log()


class PixelPopulationTransformation(PixelTransformation):
    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z / (self.pixel_building_density * self.pixel_area)

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.pixel_building_density.log() - self.pixel_area.log()

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.pixel_area

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.pixel_area.log()


class PixelPopulationDensityTransformation(PixelTransformation):
    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.pixel_building_density

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.pixel_building_density.log()

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.pixel_area

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log()


class PixelLogPopulationDensityTransformation(PixelTransformation):
    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() / self.pixel_building_density

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z - self.pixel_building_density.log()

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.pixel_area

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp()

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z


class AdminTransformation(Transformation, abc.ABC):
    """A transformation for administrative unit-level data."""

    def to_pixel_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level occupancy rate."""
        msg = "Pixel transformations are not supported for admin data"
        raise NotImplementedError(msg)

    def to_pixel_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level log occupancy rate."""
        msg = "Pixel transformations are not supported for admin data"
        raise NotImplementedError(msg)

    def to_pixel_population(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level population."""
        msg = "Pixel transformations are not supported for admin data"
        raise NotImplementedError(msg)

    def to_pixel_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level population density."""
        msg = "Pixel transformations are not supported for admin data"
        raise NotImplementedError(msg)

    def to_pixel_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from a target to pixel-level log population density."""
        msg = "Pixel transformations are not supported for admin data"
        raise NotImplementedError(msg)

    def get_transformation(
        self,
        target: ModelTarget,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        transformation = {
            ModelTarget.ADMIN_OCCUPANCY_RATE: self.to_admin_occupancy_rate,
            ModelTarget.ADMIN_LOG_OCCUPANCY_RATE: self.to_admin_log_occupancy_rate,
            ModelTarget.ADMIN_POPULATION: self.to_admin_population,
            ModelTarget.ADMIN_POPULATION_DENSITY: self.to_admin_population_density,
            ModelTarget.ADMIN_LOG_POPULATION_DENSITY: self.to_admin_log_population_density,
        }.get(target)

        if transformation is None:
            msg = f"Invalid target: {target}"
            raise ValueError(msg)
        return transformation


class AdminOccupancyRateTransformation(AdminTransformation):
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log()

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.admin_building_density * self.pixel_area

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.admin_building_density

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() + self.admin_building_density.log()


class AdminLogOccupancyRateTransformation(AdminTransformation):
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp()

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.admin_building_density * self.pixel_area

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.admin_building_density

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.admin_building_density.log()


class AdminPopulationTransformation(AdminTransformation):
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z / (self.admin_building_density * self.pixel_area)

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.admin_building_density.log() - self.pixel_area.log()

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.pixel_area

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.pixel_area.log()


class AdminPopulationDensityTransformation(AdminTransformation):
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.admin_building_density

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.log() - self.admin_building_density.log()

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.pixel_area

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.log()


class AdminLogPopulationDensityTransformation(AdminTransformation):
    def to_admin_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() / self.admin_building_density

    def to_admin_log_occupancy_rate(self, z: torch.Tensor) -> torch.Tensor:
        return z - self.admin_building_density.log()

    def to_admin_population(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp() * self.pixel_area

    def to_admin_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z.exp()

    def to_admin_log_population_density(self, z: torch.Tensor) -> torch.Tensor:
        return z
