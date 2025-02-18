import itertools
from enum import StrEnum

import numpy as np
import rasterra as rt

from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.model_prep.features import utils
from rra_population_model.model_prep.features.metadata import FeatureMetadata


def process_built_measure(
    built_version: pmc.BuiltVersion,
    measure: str,
    feature_metadata: FeatureMetadata,
    bd_data: BuildingDensityData,
    pm_data: PopulationModelData,
) -> None:
    out_measure = f"{built_version.name}_{measure}"
    strategy, fill_time_points = get_processing_strategy(
        built_version=built_version,
        time_point=feature_metadata.time_point,
    )
    if strategy == STRATEGIES.SKIP:
        print(f"Skipping {measure} for {built_version.name}.")
        return
    elif strategy == STRATEGIES.PROCESS:
        print(f"Processing {measure} for {built_version.name}.")
        source_path = bd_data.tile_path(
            provider=built_version.name,
            measure=measure,
            **feature_metadata.shared_kwargs,
        )
        if not source_path.exists():
            msg = f"Source path {source_path} does not exist."
            raise FileNotFoundError(msg)
        print(f"Linking {measure} for {built_version.name}.")
        for time_point in [feature_metadata.time_point, *fill_time_points]:
            pm_data.link_feature(
                source_path=source_path,
                feature_name=out_measure,
                resolution=feature_metadata.resolution,
                block_key=feature_metadata.block_key,
                time_point=time_point,
            )

        print(f"Mosaicking {measure} for {built_version.name}.")
        buffered_measure = mosaic_tile(
            provider=built_version.name,
            measure=measure,
            time_point=feature_metadata.time_point,
            feature_metadata=feature_metadata,
            bd_data=bd_data,
        )

    elif strategy == STRATEGIES.INTERPOLATE:
        tp_start, tp_end, w = get_time_points_and_weight(
            built_version=built_version,
            time_point=feature_metadata.time_point,
        )
        print(f"Interpolating {measure} for {built_version.name}.")
        print(
            f"Start: {tp_start}, End: {tp_end}, Weight: {w}, Time Point: {feature_metadata.time_point}"
        )
        print("Loading tiles")
        tile_start = bd_data.load_tile(
            resolution=feature_metadata.resolution,
            provider=built_version.name,
            block_key=feature_metadata.block_key,
            time_point=tp_start,
            measure=measure,
        )
        tile_end = bd_data.load_tile(
            resolution=feature_metadata.resolution,
            provider=built_version.name,
            block_key=feature_metadata.block_key,
            time_point=tp_end,
            measure=measure,
        )
        print("Interpolating and saving")
        built_measure = tile_start * w + tile_end * (1 - w)
        built_measure = utils.suppress_noise(built_measure)
        pm_data.save_feature(
            built_measure,
            feature_name=out_measure,
            **feature_metadata.shared_kwargs,
        )

        print(f"Mosaicking {measure} for {built_version.name}.")
        buffered_measure_start = mosaic_tile(
            provider=built_version.name,
            measure=measure,
            time_point=tp_start,
            feature_metadata=feature_metadata,
            bd_data=bd_data,
        )
        buffered_measure_end = mosaic_tile(
            provider=built_version.name,
            measure=measure,
            time_point=tp_end,
            feature_metadata=feature_metadata,
            bd_data=bd_data,
        )
        buffered_measure = buffered_measure_start * w + buffered_measure_end * (1 - w)
        buffered_measure = utils.suppress_noise(buffered_measure)

    for radius in pmc.FEATURE_AVERAGE_RADII:
        print(f"Processing {measure} for {built_version.name} with radius {radius}m.")
        average_measure = (
            utils.make_spatial_average(
                tile=buffered_measure,
                radius=radius,
                kernel_type="gaussian",
            )
            .resample_to(feature_metadata.block_template, "average")
            .astype(np.float32)
        )
        pm_data.save_feature(
            average_measure,
            feature_name=f"{out_measure}_{radius}m",
            **feature_metadata.shared_kwargs,
        )

        source_path = pm_data.feature_path(
            feature_name=f"{out_measure}_{radius}m",
            **feature_metadata.shared_kwargs,
        )
        for time_point in fill_time_points:
            pm_data.link_feature(
                source_path=source_path,
                feature_name=f"{out_measure}_{radius}m",
                resolution=feature_metadata.resolution,
                block_key=feature_metadata.block_key,
                time_point=time_point,
            )


def process_residential_density(
    provider: str,
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
) -> None:
    suffixes = ["", *[f"_{r}m" for r in pmc.FEATURE_AVERAGE_RADII]]
    measures = [
        f"{m}{suffix}"
        for m, suffix in itertools.product(["density", "volume"], suffixes)
    ]
    for measure in measures:
        combined = pm_data.load_feature(
            feature_name=f"{provider}_{measure}",
            **feature_metadata.shared_kwargs,
        )
        nonresidential = pm_data.load_feature(
            feature_name=f"{provider}_nonresidential_{measure}",
            **feature_metadata.shared_kwargs,
        )
        residential = combined - nonresidential
        pm_data.save_feature(
            residential,
            feature_name=f"{provider}_residential_{measure}",
            **feature_metadata.shared_kwargs,
        )


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


class STRATEGIES(StrEnum):
    PROCESS = "process"
    SKIP = "skip"
    INTERPOLATE = "interpolate"


def get_processing_strategy(
    built_version: pmc.BuiltVersion,
    time_point: str,
) -> tuple[STRATEGIES, list[str]]:
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
    bv_tps = built_version.time_points
    bv_ftps = built_version.time_points_float
    first, last = bv_tps[0], bv_tps[-1]

    year, quarter = time_point.split("q")
    time_point_float = float(year) + (float(quarter) - 1) / 4

    fill_time_points = []
    if time_point in bv_tps:
        # If the time point is in the built version, we process
        strategy = STRATEGIES.PROCESS
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
        strategy = STRATEGIES.SKIP
    else:
        # The time point is between two time points, so we need to interpolate.
        strategy = STRATEGIES.INTERPOLATE

    return strategy, fill_time_points


def mosaic_tile(
    provider: str,
    measure: str,
    time_point: str,
    feature_metadata: FeatureMetadata,
    bd_data: BuildingDensityData,
) -> rt.RasterArray:
    tiles = []
    for bk, bounds in feature_metadata.block_bounds.items():
        try:
            tile = bd_data.load_tile(
                resolution=feature_metadata.resolution,
                provider=provider,
                block_key=bk,
                time_point=time_point,
                measure=measure,
                bounds=bounds,
            )
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
            tiles.append(tile)
        except ValueError:  # noqa: PERF203
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
