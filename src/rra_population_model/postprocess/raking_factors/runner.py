from collections import defaultdict
from typing import NamedTuple

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
from rra_tools import jobmon, parallel

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import get_prediction_time_point

RAKING_VERSION = "gbd_2023"


def load_admin_populations(
    pm_data: PopulationModelData,
    time_point: str,
) -> gpd.GeoDataFrame:
    raking_pop = pm_data.load_raking_population(version=RAKING_VERSION)
    all_pop = raking_pop.loc[raking_pop.most_detailed == 1].set_index(
        ["year_id", "location_id"]
    )["population"]

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

    admins = pm_data.load_raking_shapes(version=RAKING_VERSION)
    pop = admins[["location_id", "geometry"]].merge(pop, on="location_id")
    return pop


class AggregationArgs(NamedTuple):
    resolution: str
    model_version: str
    block_key: str
    time_point: str
    shape_map: dict[int, shapely.Polygon]
    tile_poly: shapely.Polygon


def aggregate_unraked_population(
    aggregate_args: AggregationArgs,
) -> dict[int, float]:
    est_pop: dict[int, float] = {}
    resolution, model_version, block_key, time_point, shape_map, *_ = aggregate_args
    if not shape_map:
        return est_pop

    pm_data = PopulationModelData()
    model_spec = pm_data.load_model_specification(resolution, model_version)

    r = pm_data.load_raw_prediction(block_key, time_point, model_spec)
    for location_id, geom in shape_map.items():
        est_pop[location_id] = np.nansum(r.mask(geom))  # type: ignore[assignment]
    return est_pop


def raking_factors_main(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("loading model frame")
    model_spec = pm_data.load_model_specification(resolution, version)
    model_frame = pm_data.load_modeling_frame(resolution)
    population = load_admin_populations(pm_data, time_point)
    population = population.to_crs(model_frame.crs)
    prediction_time_point = get_prediction_time_point(
        pm_data, resolution, version, time_point
    )

    print("Building location aggregation args")
    block_keys = model_frame.block_key.unique().tolist()
    compute_location_args = []
    for block_key in tqdm.tqdm(block_keys, disable=not progress_bar):
        tile_poly = shapely.box(
            *model_frame[model_frame.block_key == block_key].total_bounds
        )
        shape_map = (
            population[population.intersects(tile_poly)]
            .clip(tile_poly)
            .set_index("location_id")
            .geometry.to_dict()
        )
        compute_location_args.append(
            AggregationArgs(
                resolution,
                version,
                block_key,
                prediction_time_point,
                shape_map,
                tile_poly,
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
    combined = pd.concat([aggregate_pop.rename("raw"), loc_pop.rename("true")], axis=1)
    combined["raking_factor"] = combined["true"] / combined["raw"]

    print("Building raking args")
    raking_factors_by_block = []
    for loc_args in tqdm.tqdm(compute_location_args, disable=not progress_bar):
        for location_id, geom in loc_args.shape_map.items():
            raw, true, rf = combined.loc[location_id].tolist()  # type: ignore[operator]
            raking_factors_by_block.append(
                (loc_args.block_key, location_id, raw, true, rf, geom)
            )

    print("Collating")
    final_raking_factors = gpd.GeoDataFrame(
        raking_factors_by_block,
        columns=[
            "block_key",
            "location_id",
            "raw_pop",
            "true_pop",
            "raking_factor",
            "geometry",
        ],
        crs=model_frame.crs,
    )
    print("Saving")
    pm_data.save_raking_factors(final_raking_factors, time_point, model_spec)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_progress_bar()
def raking_factors_task(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    raking_factors_main(
        resolution, version, time_point, output_dir, num_cores, progress_bar
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_copy_from_version()
@clio.with_time_point(choices=None, allow_all=True)
@click.option("--extrapolate", is_flag=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_queue()
def raking_factors(
    resolution: str,
    version: str,
    copy_from_version: str | None,
    time_point: str,
    extrapolate: bool,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    pm_data.maybe_copy_version(resolution, version, copy_from_version)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = clio.convert_choice(time_point, prediction_time_points)
    if extrapolate:
        full_time_series = [f"{y}q1" for y in range(1950, 2101)]
        time_points = sorted(set(time_points) | set(full_time_series))

    print(f"Building raking factors for {len(time_points)} time points.")

    print("Generating raking factors")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="raking_factors",
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": f"{num_cores * 15}G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        node_args={
            # "version": [f"2025_06_21.00{x}" for x in range(1, 5)],
            "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "num-cores": num_cores,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_raking_factors"),
    )
