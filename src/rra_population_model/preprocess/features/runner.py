import itertools
from pathlib import Path

import click
import numpy as np
import rasterra as rt
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.preprocess.features import (
    utils,
)


def mosaic_tile(
    feature_spec: tuple[str, str, str],
    feature_metadata: utils.FeatureMetadata,
    bd_data: BuildingDensityData,
) -> rt.RasterArray:
    provider, in_measure, _ = feature_spec
    tiles = []
    for bk, bounds in feature_metadata.block_bounds.items():
        try:
            tile = bd_data.load_tile(
                resolution=feature_metadata.resolution,
                provider=provider,
                block_key=bk,
                time_point=feature_metadata.time_point,
                measure=in_measure,
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


def process_bd_feature(
    feature_spec: tuple[str, str, str],
    feature_metadata: utils.FeatureMetadata,
    fill_time_points: list[str],
    bd_data: BuildingDensityData,
    pm_data: PopulationModelData,
) -> None:
    provider, in_measure, out_measure = feature_spec
    print(f"Processing {out_measure} for {provider}")

    source_path = bd_data.tile_path(
        provider=provider,
        measure=in_measure,
        **feature_metadata.shared_kwargs,
    )
    if not source_path.exists():
        msg = f"Source path {source_path} does not exist."
        raise FileNotFoundError(msg)

    for time_point in [feature_metadata.time_point, *fill_time_points]:
        pm_data.link_feature(
            source_path=source_path,
            feature_name=out_measure,
            resolution=feature_metadata.resolution,
            block_key=feature_metadata.block_key,
            time_point=time_point,
        )

    buffered_measure = mosaic_tile(
        feature_spec=feature_spec,
        feature_metadata=feature_metadata,
        bd_data=bd_data,
    )

    for radius in pmc.FEATURE_AVERAGE_RADII:
        print(f"Processing {out_measure} for {provider} with radius {radius}m.")
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


def get_allowed_and_fill_time_points(
    provider: str, time_point: str
) -> tuple[bool, list[str]]:
    allowed_time_points = {
        **pmc.MICROSOFT_TIME_POINTS,
        "ghsl_r2023a": pmc.ALL_TIME_POINTS,
    }[provider]

    allowed = time_point in allowed_time_points
    if allowed:
        # Check if our time point is on the edge of the allowed time points,
        # and if so, add all prior or subsequent time points to the fill list.
        if time_point == allowed_time_points[0]:
            fill_time_points = pmc.ALL_TIME_POINTS[
                : pmc.ALL_TIME_POINTS.index(time_point)
            ]
        elif time_point == allowed_time_points[-1]:
            fill_time_points = pmc.ALL_TIME_POINTS[
                pmc.ALL_TIME_POINTS.index(time_point) + 1 :
            ]
        else:
            fill_time_points = []
    else:
        fill_time_points = []
    return allowed, fill_time_points


def features_main(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str | Path,
    model_root: str | Path,
) -> None:
    print(f"Processing features for block {block_key} at time {time_point}")
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(model_root)

    print("Loading all feature metadata")
    feature_metadata = utils.get_feature_metadata(
        pm_data, bd_data, resolution, block_key, time_point
    )

    providers_and_measures = [
        ("microsoft_v2", "density", "msftv2_density"),
        ("microsoft_v3", "density", "msftv3_density"),
        ("microsoft_v4", "density", "msftv4_density"),
        ("ghsl_r2023a", "density", "ghsl_density"),
        ("ghsl_r2023a", "nonresidential_density", "ghsl_nonresidential_density"),
        ("ghsl_r2023a", "volume", "ghsl_volume"),
        ("ghsl_r2023a", "nonresidential_volume", "ghsl_nonresidential_volume"),
    ]
    for feature_spec in providers_and_measures:
        allowed, fill_time_points = get_allowed_and_fill_time_points(
            provider=feature_spec[0],
            time_point=time_point,
        )
        if allowed:
            process_bd_feature(
                feature_spec=feature_spec,
                feature_metadata=feature_metadata,
                fill_time_points=fill_time_points,
                bd_data=bd_data,
                pm_data=pm_data,
            )

    allowed, fill_time_points = get_allowed_and_fill_time_points(
        "ghsl_r2023a", time_point
    )
    assert (  # noqa: PT018, S101
        allowed and not fill_time_points
    ), "GHSL r2023a should not need fill time points."

    print("Processing residential density")
    suffixes = ["", *[f"_{r}m" for r in pmc.FEATURE_AVERAGE_RADII]]
    measures = [
        f"{m}{suffix}"
        for m, suffix in itertools.product(["density", "volume"], suffixes)
    ]
    for measure in measures:
        combined = pm_data.load_feature(
            feature_name=f"ghsl_{measure}",
            **feature_metadata.shared_kwargs,
        )
        nonresidential = pm_data.load_feature(
            feature_name=f"ghsl_nonresidential_{measure}",
            **feature_metadata.shared_kwargs,
        )
        residential = combined - nonresidential
        pm_data.save_feature(
            residential,
            feature_name=f"ghsl_residential_{measure}",
            **feature_metadata.shared_kwargs,
        )

    print("Processing NTL")
    ntl = utils.load_and_format_ntl(feature_metadata)
    pm_data.save_feature(
        ntl,
        feature_name="nighttime_lights",
        **feature_metadata.shared_kwargs,
    )
    log_ntl = np.log(1 + ntl)
    pm_data.save_feature(
        log_ntl,
        feature_name="log_nighttime_lights",
        **feature_metadata.shared_kwargs,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def features_task(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    """Build predictors for a given tile and time point."""
    features_main(block_key, time_point, resolution, building_density_dir, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_time_point(allow_all=True)
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def features(
    time_point: list[str],
    resolution: str,
    building_density_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    """Prepare model features."""
    pm_data = PopulationModelData(output_dir)
    print("Loading the modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()
    block_times = list(itertools.product(block_keys, time_point))

    jobmon.run_parallel(
        task_name="features",
        flat_node_args=(("block-key", "time-point"), block_times),
        task_args={
            "building-density-dir": building_density_dir,
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "15G",
            "runtime": "15m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        runner="pmtask preprocess",
        log_root=pm_data.log_dir("preprocess_features"),
        max_attempts=1,
    )
