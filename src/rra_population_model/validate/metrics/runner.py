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

    print("Loading validation frame")
    validation_frame = pm_data.load_validation_frame(resolution)
    block_validation_frame = validation_frame.loc[
        validation_frame.block_key == block_key
    ]
    validation_locs = list(
        block_validation_frame[["iso3", "year"]].itertuples(index=False)
    )

    print("Loading and subsetting modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_frame = (
        modeling_frame.loc[
            modeling_frame.block_key == block_key, ["block_key", "geometry"]
        ]
        .dissolve("block_key")
        .reset_index()
    )
    block_poly = block_frame.geometry.iloc[0]

    print("Loading and raked population predictions")
    model_spec = pm_data.load_model_specification(resolution, version)
    pop_raster = pm_data.load_raked_prediction(block_key, time_point, model_spec)
    pop_arr = pop_raster._ndarray  # noqa: SLF001

    print("Loading and subsetting census data")
    iter_data = []
    for iso3, year in validation_locs:
        path = pm_data.census_path(iso3, year)
        max_admin_level = int(
            pd.read_parquet(path, columns=["admin_level"]).admin_level.max()
        )
        gdf = gpd.read_parquet(
            path,
            bbox=block_poly.bounds,
            filters=[("admin_level", "==", max_admin_level)],
        )
        gdf = gdf[gdf.buffer(0).intersects(block_poly)]
        if not gdf.empty:
            iter_data.append((iso3, year, gdf))

    if not iter_data:
        print("No census data found for block", block_key)
        return

    print("Calculating pixel metrics")
    out = []
    for iso3, year, gdf in iter_data:
        shape_values = [(shape, i + 1) for i, shape in enumerate(gdf.geometry)]
        bounds_map = build_bounds_map(pop_raster, shape_values)

        location_mask = np.zeros_like(pop_raster, dtype=np.uint32)
        location_mask = rasterize(
            shape_values,
            out=location_mask,
            transform=pop_raster.transform,
            merge_alg=MergeAlg.replace,
        )
        final_bounds_map = {
            i - 1: (rows, cols, location_mask[rows, cols] == i)
            for i, (rows, cols) in bounds_map.items()
        }

        data = []
        for rows, cols, mask in final_bounds_map.values():
            loc_pop = np.nansum(pop_arr[rows, cols][mask])
            data.append(loc_pop)

        loc_results = gdf[["shape_id"]].copy()
        loc_results["iso3"] = iso3
        loc_results["year"] = year
        loc_results["population"] = data
        out.append(loc_results)

    print("Saving results")
    results = pd.concat(out)
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

    time_points = pm_data.list_raking_factor_time_points(resolution, version)
    if time_point not in time_points:
        msg = (
            f"Time point {time_point} not found in {resolution} {version}.\n"
            f"Valid time points are: {time_points}"
        )
        raise ValueError(msg)

    validation_frame = pm_data.load_validation_frame(resolution)
    block_keys = list(validation_frame.block_key.unique())

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
            "version": [f"2025_04_24.00{i}" for i in range(1, 9)],
        },
        task_args={
            "resolution": resolution,
            "output-dir": output_dir,
            "time-point": time_point,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("validate_pixel_metrics"),
    )
