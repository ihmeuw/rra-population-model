import abc
from pathlib import Path

import numpy as np
import rasterra as rt

from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.model_prep.features import utils
from rra_population_model.model_prep.features.metadata import FeatureMetadata


class ProcessingStrategy:
    def __init__(
        self,
        built_version: pmc.BuiltVersion,
        feature_metadata: FeatureMetadata,
    ) -> None:
        self.built_version = built_version
        self.feature_metadata = feature_metadata

    @abc.abstractmethod
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def generate_derived_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def generate_geospatial_averages(
        self,
        features: list[str],
        feature_average_radii: list[int],
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def link_features(
        self,
        feature_paths: dict[str, Path],
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,
        pm_data: PopulationModelData,
    ) -> None:
        pass


class SkipStrategy(ProcessingStrategy):
    def generate_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def generate_derived_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def generate_geospatial_averages(
        self,
        features: list[str],  # noqa: ARG002
        feature_average_radii: list[int],  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def link_features(
        self,
        feature_paths: dict[str, Path],  # noqa: ARG002
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> None:
        assert fill_time_points == []  # noqa: S101
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )


class ProcessStrategy(ProcessingStrategy):
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        out_paths = {}
        for measure in self.built_version.measures:
            out_measure = f"{self.built_version.name}_{measure}"
            print(f"Processing {measure} for {self.built_version.name}.")
            source_path = bd_data.tile_path(
                provider=self.built_version.name,
                measure=measure,
                **self.feature_metadata.shared_kwargs,
            )
            if not source_path.exists():
                msg = f"Source path {source_path} does not exist."
                raise FileNotFoundError(msg)
            print(f"Linking {measure} for {self.built_version.name}.")
            pm_data.link_feature(
                source_path=source_path,
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
            out_paths[out_measure] = source_path
        return out_paths

    def generate_derived_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        if self.built_version.name == "ghsl_r2023a":
            # No derived measures for GHSL
            return {}
        elif self.built_version.name.startswith("microsoft"):
            return _generate_microsoft_derived_measures(
                pm_data, self.feature_metadata, self.built_version.name
            )
        else:
            msg = f"Unknown built version: {self.built_version.name}"
            raise ValueError(msg)

    def generate_geospatial_averages(
        self,
        features: list[str],
        feature_average_radii: list[int],
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        out_paths = {}
        for feature in features:
            for radius in feature_average_radii:
                print(f"Processing {feature} with radius {radius}m.")
                buffered_measure = mosaic_tile(
                    measure=feature,
                    feature_metadata=self.feature_metadata,
                    pm_data=pm_data,
                )
                print(f"Processing {feature} with radius {radius}m.")
                average_measure = (
                    utils.make_spatial_average(
                        tile=buffered_measure,
                        radius=radius,
                        kernel_type="gaussian",
                    )
                    .resample_to(self.feature_metadata.block_template, "average")
                    .astype(np.float32)
                )
                pm_data.save_feature(
                    average_measure,
                    feature_name=f"{feature}_{radius}m",
                    **self.feature_metadata.shared_kwargs,
                )

                out_paths[f"{feature}_{radius}m"] = pm_data.feature_path(
                    feature_name=f"{feature}_{radius}m",
                    **self.feature_metadata.shared_kwargs,
                )
        return out_paths

    def link_features(
        self,
        feature_paths: dict[str, Path],
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,
        pm_data: PopulationModelData,
    ) -> None:
        for feature, path in feature_paths.items():
            for time_point in fill_time_points:
                pm_data.link_feature(
                    source_path=path,
                    feature_name=feature,
                    time_point=time_point,
                    block_key=feature_metadata.block_key,
                    resolution=feature_metadata.resolution,
                )


class InterpolateStrategy(ProcessStrategy):
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        out_paths = {}
        for measure in self.built_version.measures:
            out_measure = f"{self.built_version.name}_{measure}"
            tp_start, tp_end, w = get_time_points_and_weight(
                built_version=self.built_version,
                time_point=self.feature_metadata.time_point,
            )
            print(f"Interpolating {measure} for {self.built_version.name}.")
            print(
                f"Start: {tp_start}, End: {tp_end}, Weight: {w}, Time Point: {self.feature_metadata.time_point}"
            )
            print("Loading tiles")
            tile_start = bd_data.load_tile(
                resolution=self.feature_metadata.resolution,
                provider=self.built_version.name,
                block_key=self.feature_metadata.block_key,
                time_point=tp_start,
                measure=measure,
            )
            tile_end = bd_data.load_tile(
                resolution=self.feature_metadata.resolution,
                provider=self.built_version.name,
                block_key=self.feature_metadata.block_key,
                time_point=tp_end,
                measure=measure,
            )
            print("Interpolating and saving")
            built_measure = tile_start * w + tile_end * (1 - w)
            built_measure = utils.suppress_noise(built_measure)
            pm_data.save_feature(
                built_measure,
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
            out_paths[out_measure] = pm_data.feature_path(
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
        return out_paths

    def link_features(
        self,
        feature_paths: dict[str, Path],  # noqa: ARG002
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> None:
        assert fill_time_points == []  # noqa: S101
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )


HEIGHT_MIN = 2.4384  # 8ft


def _generate_microsoft_derived_measures(
    pm_data: PopulationModelData,
    feature_metadata: FeatureMetadata,
    built_version_name: str,
) -> dict[str, Path]:
    feature_dict = {
        "microsoft_v6": {
            "density": "microsoft_v6_density",
            "height": "ghsl_r2023a_height",
            "p_residential": "ghsl_r2023a_proportion_residential",
            "reference_density": "ghsl_r2023a_density",
        },
        "microsoft_v7": {
            "density": "microsoft_v7_density",
            "height": "microsoft_v7_height",
            "p_residential": "ghsl_r2023a_proportion_residential",
            "reference_density": "ghsl_r2023a_density",
        },
        "microsoft_v7_1": {
            "density": "microsoft_v7_1_density",
            "height": "microsoft_v7_1_height",
            "p_residential": "ghsl_r2023a_proportion_residential",
            "reference_density": "ghsl_r2023a_density",
        },
    }[built_version_name]
    density = pm_data.load_feature(
        feature_name=feature_dict["density"],
        **feature_metadata.shared_kwargs,
    )
    density_arr = density._ndarray  # noqa: SLF001
    height_arr = pm_data.load_feature(  # noqa: SLF001
        feature_name=feature_dict["height"],
        **feature_metadata.shared_kwargs,
    )._ndarray
    p_residential_arr = pm_data.load_feature(  # noqa: SLF001
        feature_name=feature_dict["p_residential"],
        **feature_metadata.shared_kwargs,
    )._ndarray

    # CROSSWALKING PROCEDURES
    # 1) ensure we have height wherever there is density, even if GHSL doesn't think there is density
    if feature_dict["density"].replace("_density", "") != feature_dict["height"].replace("_height", ""):
        height_min = HEIGHT_MIN
        if (height_arr > 0).any():
            height_min = float(np.nanmin(height_arr[height_arr > 0]))
        density_threshold = 0.01
        density_is_positive = density_arr >= density_threshold
        height_is_zero = height_arr == 0
        height_arr[density_is_positive & height_is_zero] = height_min

    # 2) if Microsoft places buildings somewhere GHSL does not, call them residential
    if "reference_density" in feature_dict.keys():
        reference_density_arr = pm_data.load_feature(
            feature_name=feature_dict["reference_density"],
            **feature_metadata.shared_kwargs,
        )._ndarray
        density_threshold = 0.01
        density_is_positive = density_arr >= density_threshold
        reference_density_is_zero = reference_density_arr < density_threshold
        residential_is_zero = p_residential_arr == 0
        p_residential_arr[density_is_positive & reference_density_is_zero & residential_is_zero] = 1

    out_ops = {
        "density": lambda d, _, __: d,
        "residential_density": lambda d, _, p: d * p,
        "nonresidential_density": lambda d, _, p: d * (1 - p),
        "volume": lambda d, h, _: h * d,
        "residential_volume": lambda d, h, p: h * d * p,
        "nonresidential_volume": lambda d, h, p: h * d * (1 - p),
    }
    for measure, op in out_ops.items():
        out = rt.RasterArray(
            data=op(density_arr, height_arr, p_residential_arr),  # type: ignore[no-untyped-call]
            transform=density.transform,
            crs=density.crs,
            no_data_value=np.nan,
        )
        pm_data.save_feature(
            out,
            feature_name=f"{built_version_name}_{measure}",
            **feature_metadata.shared_kwargs,
        )

    out_paths = {
        f"{built_version_name}_{m}": pm_data.feature_path(
            feature_name=f"{built_version_name}_{m}",
            **feature_metadata.shared_kwargs,
        )
        for m in out_ops
    }
    return out_paths


def get_time_points_and_weight(
    built_version: pmc.BuiltVersion,
    time_point: str,
) -> tuple[str, str, float]:
    year, quarter = time_point.split("q")
    time_point_float = float(year) + (float(quarter) - 1) / 4
    bv_ftps = built_version.time_points_float

    tp_start, tp_end, w = "", "", 0.0
    for i, (t_start, t_end) in enumerate(zip(bv_ftps[:-1], bv_ftps[1:], strict=False)):
        if t_start <= time_point_float <= t_end:
            tp_start = built_version.time_points[i]
            tp_end = built_version.time_points[i + 1]
            w = (t_end - time_point_float) / (t_end - t_start)
    return tp_start, tp_end, w


def get_processing_strategy(
    built_version: pmc.BuiltVersion,
    feature_metadata: FeatureMetadata,
) -> tuple[ProcessingStrategy, list[str]]:
    """Determine the processing strategy for a given time point.

    This method is used to flexibly determine a processing strategy for a given time
    point. The strategy is determined based on the time point and the built version.
    The strategy is one of the following:

        - "process": The time point is in the built version. We process normally. If
            the time point is also terminal, we will also fill in the time points
            before and after the time point.
        - "skip": The time point is beyond the terminal time points of the built
            version. In this case, we will skip the time point as it will be extrapolated
            when we process the terminal time point.
        - "interpolate": The time point is between two time points in the built version.
            In this case, we will interpolate the time point and process it.

    """
    time_point = feature_metadata.time_point
    bv_tps = built_version.time_points
    bv_ftps = built_version.time_points_float
    first, last = bv_tps[0], bv_tps[-1]

    year, quarter = time_point.split("q")
    time_point_float = float(year) + (float(quarter) - 1) / 4

    strategy: ProcessingStrategy
    fill_time_points = []
    if time_point in bv_tps:
        # If the time point is in the built version, we process
        strategy = ProcessStrategy(
            built_version=built_version, feature_metadata=feature_metadata
        )
        # If the time point is also terminal, we extrapolate as well
        # A time point can be first, last, or both (i.e. if the version has a
        # single time point)
        if time_point == first:
            fill_time_points.extend(
                pmc.ALL_TIME_POINTS[: pmc.ALL_TIME_POINTS.index(first)]
            )
        if time_point == last:
            fill_time_points.extend(
                pmc.ALL_TIME_POINTS[pmc.ALL_TIME_POINTS.index(last) + 1 :]
            )
    elif time_point_float < bv_ftps[0] or time_point_float > bv_ftps[-1]:
        # If the time point is before the first or after the last time point,
        # it will be extrapolated when we process the terminal time point,
        # so we don't need to do anything here.
        strategy = SkipStrategy(
            built_version=built_version, feature_metadata=feature_metadata
        )
    else:
        # The time point is between two time points, so we need to interpolate.
        strategy = InterpolateStrategy(
            built_version=built_version, feature_metadata=feature_metadata
        )

    return strategy, fill_time_points


def mosaic_tile(
    measure: str,
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
) -> rt.RasterArray:
    tiles = []
    for bounds in feature_metadata.block_bounds.values():
        try:
            tile = pm_data.load_feature(
                feature_name=measure,
                **feature_metadata.shared_kwargs,
                subset_bounds=bounds,
            )
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
            tiles.append(tile)
        except ValueError:
            # This is kind of a hack, but there's not a clean way to fix it easily.
            # The issue is that the resolution of the tiles do not exactly line up
            # to the bounds of the world as defined by the CRS. The southernmost
            # have one fewer row of pixels as a whole row would extend past the
            # southern edge of the world, causing reprojection issues. The problem
            # is that we read the tile with bounds, and those bounds cause the underlying
            # rasterio to fill in that missing row. Here we just remove the last row
            # of pixels and reproject the tile.
            tile._ndarray = tile._ndarray[:-1].copy()  # noqa: SLF001
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
            tiles.append(tile)

    buffered_measure = utils.suppress_noise(rt.merge(tiles))
    return buffered_measure
