from collections import defaultdict
from typing import NamedTuple

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import shapely
import tqdm
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon, parallel
from rra_tools.shell_tools import mkdir, touch

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData


def load_admin_populations(
    pm_data: PopulationModelData,
    time_point: str,
) -> gpd.GeoDataFrame:
    raking_pop = pm_data.load_raking_population(version="fhs_2021_wpp_2022")
    all_pop = raking_pop.set_index(["year_id", "location_id"])["population"]

    # Interpolate the time point population
    if "q" in time_point:
        year, quarter = (int(s) for s in time_point.split("q"))
        next_year = min(year + 1, 2100)
        weight = (int(quarter) - 1) / 4

        prior_year_pop = all_pop.loc[year]
        next_year_pop = all_pop.loc[next_year]

        pop = ((1 - weight) * prior_year_pop + weight * next_year_pop).reset_index()
    else:
        year = int(time_point)
        pop = all_pop.loc[year]

    admins = pm_data.load_raking_shapes(version="fhs_2021_wpp_2022")
    pop = admins[["location_id", "geometry"]].merge(pop, on="location_id")
    return pop


class AggregationArgs(NamedTuple):
    resolution: str
    model_name: str
    block_key: str
    time_point: str
    shape_map: dict[int, shapely.Polygon]
    tile_poly: shapely.Polygon


def get_load_time_point(time_point: str) -> str:
    if "q" in time_point:
        return time_point
    else:
        load_year = min(max(int(time_point), 1975), 2023)
        return f"{load_year}q1"


def aggregate_unraked_population(
    aggregate_args: AggregationArgs,
) -> dict[int, float]:
    est_pop: dict[int, float] = {}
    resolution, model_name, block_key, time_point, shape_map, *_ = aggregate_args
    if not shape_map:
        return est_pop

    pm_data = PopulationModelData()
    model_spec = pm_data.load_model_specification(resolution, model_name)
    load_time_point = get_load_time_point(time_point)

    r = pm_data.load_prediction(block_key, load_time_point, model_spec)
    for location_id, geom in shape_map.items():
        est_pop[location_id] = np.nansum(r.mask(geom))  # type: ignore[assignment]
    return est_pop


def make_raking_factors_main(
    resolution: str,
    model_name: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("loading model frame")
    model_frame = pm_data.load_modeling_frame(resolution)
    population = load_admin_populations(pm_data, time_point)
    population = population.to_crs(model_frame.crs)

    print("Building location aggregation args")
    block_keys = model_frame.block_key.unique().tolist()
    compute_location_args = []
    for block_key in tqdm.tqdm(block_keys, disable=not progress_bar):
        tile_poly = shapely.box(
            *model_frame[model_frame.block_key == block_key].total_bounds
        )
        shape_map = (
            population[population.intersects(tile_poly)]
            .set_index("location_id")
            .geometry.to_dict()
        )
        compute_location_args.append(
            AggregationArgs(
                resolution, model_name, block_key, time_point, shape_map, tile_poly
            )
        )

    print("Aggregating population")
    aggregate_pops_by_block = parallel.run_parallel(
        aggregate_unraked_population,
        compute_location_args,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    print("Collating aggregate population across blocks")
    aggregate_pops: dict[int, float] = defaultdict(float)
    for pop_dict in aggregate_pops_by_block:
        for location_id, lpop in pop_dict.items():
            aggregate_pops[location_id] += lpop

    print("Calculating raking factors")
    aggregate_pop = pd.Series(aggregate_pops)
    loc_pop = population.set_index("location_id").population
    raking_factors = (loc_pop / aggregate_pop.loc[loc_pop.index]).to_dict()

    print("Building raking args")
    raking_factors_by_block = []
    for loc_args in tqdm.tqdm(compute_location_args, disable=not progress_bar):
        for location_id, geom in loc_args.shape_map.items():
            raking_factors_by_block.append(
                (loc_args.block_key, location_id, geom, raking_factors[location_id])
            )

    print("Collating")
    out = gpd.GeoDataFrame(
        raking_factors_by_block,
        columns=["block_key", "location_id", "geometry", "raking_factor"],
        crs=model_frame.crs,
    )
    print("Saving")
    out_path = (
        pm_data.raking_utility_data  # type: ignore[attr-defined]
        / resolution
        / model_name
        / f"raking_factors_{time_point}.parquet"
    )
    if out_path.exists():
        out_path.unlink()
    mkdir(out_path.parent, exist_ok=True, parents=True)
    touch(out_path)
    out.to_parquet(out_path)


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_model_name()
@clio.with_time_point(choices=pmc.ALL_TIME_POINTS)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_progress_bar()
def make_raking_factors_task(
    resolution: str,
    model_name: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    make_raking_factors_main(
        resolution, model_name, time_point, output_dir, num_cores, progress_bar
    )


def rake_main(
    resolution: str,
    model_name: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Loading metadata")
    model_frame = pm_data.load_modeling_frame(resolution)
    model_spec = pm_data.load_model_specification(resolution, model_name)
    load_time_point = get_load_time_point(time_point)
    print("Loading unraked prediction")
    unraked_data = pm_data.load_prediction(block_key, load_time_point, model_spec)

    print("Loading raking factors")
    raking_data = gpd.read_parquet(
        pm_data.raking_utility_data  # type: ignore[attr-defined]
        / resolution
        / model_name
        / f"raking_factors_{time_point}.parquet",
        filters=[("block_key", "==", block_key)],
    )

    print("Loading raking shapes")
    bounds = model_frame[model_frame.block_key == block_key].total_bounds
    tile_poly = shapely.box(*bounds)
    shapes = pm_data.load_raking_shapes(
        version="fhs_2021_wpp_2022", bbox=tile_poly.bounds
    )

    print("Raking")
    if shapes.empty:
        raking_factor = rt.RasterArray(
            np.nan * np.ones_like(unraked_data),
            transform=unraked_data.transform,
            crs=unraked_data.crs,
            no_data_value=np.nan,
        )
        raked = raking_factor
    else:
        raking_factor_data = np.nan * np.ones_like(unraked_data)
        for geom, rf in raking_data[["geometry", "raking_factor"]].itertuples(
            index=False
        ):
            shape_mask, *_ = raster_geometry_mask(
                data_transform=unraked_data.transform,
                data_width=unraked_data.shape[1],
                data_height=unraked_data.shape[0],
                shapes=[geom],
                invert=True,
            )
            raking_factor_data[shape_mask] = rf

        raking_factor = rt.RasterArray(
            raking_factor_data,
            transform=unraked_data.transform,
            crs=unraked_data.crs,
            no_data_value=np.nan,
        )
        raked = unraked_data * raking_factor

    print("Saving raked prediction")
    pm_data.save_raked_prediction(raked, block_key, time_point, model_spec)


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_model_name()
@clio.with_block_key()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_task(
    resolution: str,
    model_name: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_main(resolution, model_name, block_key, time_point, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_resolution(allow_all=False)
@clio.with_model_name()
@clio.with_time_point(allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_queue()
def rake(
    resolution: str,
    model_name: str,  # noqa: ARG001
    time_point: str,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Preparing runs")

    model_names = [
        "msftv4_density_log_ntl",
        "ghsl_density_log_ntl",
        "ghsl_volume_log_ntl",
        "ghsl_residential_density_log_ntl",
        "ghsl_residential_volume_log_ntl",
    ]

    print("Generating raking factors")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="make_raking_factors",
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": f"{num_cores * 15}G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "time-point": time_point,
            "model-name": model_names,
        },
        task_args={
            "resolution": resolution,
            "num-cores": num_cores,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.raking,
    )

    model_frame = pm_data.load_modeling_frame(resolution)
    block_keys = model_frame.block_key.unique().tolist()

    print("Raking")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "50G",
            "runtime": "25m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "block-key": block_keys,
            "time-point": time_point,
            "model-name": ["ghsl_residential_volume_log_ntl"],
        },
        task_args={
            "resolution": resolution,
            "output-dir": output_dir,
        },
        max_attempts=3,
        log_root=pm_data.raking,
    )
