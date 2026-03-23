import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import shapely
from rasterio.features import MergeAlg, rasterize
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData


def build_bounds_map(
    raster_template: rt.RasterArray,
    shape_values: list[tuple[shapely.Polygon | shapely.MultiPolygon, int]],
) -> dict[int, tuple[slice, slice]]:
    # The tranform maps pixel coordinates to the CRS coordinates.
    # This mask is the inverse of that transform.
    to_pixel = ~raster_template.transform

    bounds_map = {}
    for shp, loc_id in shape_values:
        xmin, ymin, xmax, ymax = shp.bounds
        pxmin, pymin = to_pixel * (xmin, ymax)
        pixel_buffer = 10
        pxmin = max(0, int(pxmin) - pixel_buffer)
        pymin = max(0, int(pymin) - pixel_buffer)
        pxmax, pymax = to_pixel * (xmax, ymin)
        pxmax = min(raster_template.width, int(pxmax) + pixel_buffer)
        pymax = min(raster_template.height, int(pymax) + pixel_buffer)
        bounds_map[loc_id] = (slice(pymin, pymax), slice(pxmin, pxmax))

    return bounds_map


def pixel_metrics_main(
    block_key: str,
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)


    pm_data.save_raw_validation_data(
        results,
        resolution=resolution,
        version=version,
        block_key=block_key,
        time_point=time_point,
    )


@click.command()
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
def pixel_metrics_task(
    block_key: str, time_point: str, resolution: str, version: str, output_dir: str
) -> None:
    pixel_metrics_main(block_key, time_point, resolution, version, output_dir)


@click.command()
@clio.with_time_point(choices=None)
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def metrics(
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    time_points = pm_data.list_raked_prediction_time_points(resolution, version)
    time_points = [time_point for time_point in time_points if time_point.endswith("q1")]
    if time_point not in time_points:
        msg = (
            f"Time point {time_point} not found in {resolution} {version}.\n"
            f"Valid time points are: {time_points}"
        )
        raise ValueError(msg)

    validation_frame = pm_data.load_validation_frame(resolution)
    block_keys = list(validation_frame.block_key.unique())

    # versions = [f"2025_11_08.0{(i + 1):02d}" for i in range(60)]

    jobmon.run_parallel(
        runner="pmtask validate",
        task_name="pixel_metrics",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "15G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "block-key": block_keys,
            # "version": versions,
            # "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
            "time-point": time_point,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("validate_pixel_metrics"),
    )
