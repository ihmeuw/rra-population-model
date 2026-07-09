from pathlib import Path
from loguru import logger
import functools

import click
from rra_tools import jobmon, parallel

import numpy as np
import pandas as pd
import geopandas as gpd

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.postprocess.census_rake import utils

# 0 = don't downsample; 1+ = number of downsampled admin-years to run
DOWNSAMPLE_ADMINS = 0


def census_rake_main(
    resolution: str,
    version: str,
    iso3: str,
    census_time_point: str,
    task_parent_id: str,
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
    if verbose:
        logger.info("Starting")
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    if verbose:
        logger.info("Loading census inputs")
    task_admins, census_weights = pm_data.load_census_raking_inputs(
        model_spec,
        iso3=iso3,
        census_time_point=census_time_point,
        task_parent_id=task_parent_id,
    )
    task_map = task_admins.loc[[iso3], [census_time_point], [task_parent_id], :].reset_index().T[0].rename("task_arg")
    census_weights = census_weights.loc[iso3, :, census_time_point]["weight"]
    model_time_points = sorted(census_weights.index.to_list())

    if verbose:
        logger.info(
            "Processing census and prediction data\n"
            f'    Number of admins: {task_map["most_detailed_units"]:,}\n'
            f'    Population: {int(task_map["population_total"]):,}\n'
            f'    Area: {int(task_map["area"] / 1e6):,} km^2\n'
            f'    Bounding Box Area: {int(task_map["bounds_area"] / 1e6):,} km^2'
        )
    overlay_skeleton, prediction_data, template_raster = utils.process_census_data(
        resolution,
        version,
        pm_data,
        task_map,
        model_time_points,
    )

    if prediction_data is not None and overlay_skeleton is not None:
        if verbose:
            logger.info("Raking")
        raked_rasters = utils.rake(
            overlay_skeleton,
            prediction_data,
            template_raster,
            census_time_point,
            model_time_points,
        )
        if verbose:
            logger.info("Saving raked rasters")
        for model_time_point, raked_raster in zip(model_time_points, raked_rasters):
            pm_data.save_raked_census(
                raked_raster,
                iso3,
                task_parent_id,
                model_time_point,
                census_time_point,
                model_spec,
            )
    else:
        # No population: save the template for every time point. For an admin with
        # no predicted pixels this is a minimal all-nodata raster (see
        # process_census_data); otherwise a zero raster over the valid pixels.
        if verbose:
            logger.info("Saving rasters (no population)")
        for model_time_point in model_time_points:
            pm_data.save_raked_census(
                template_raster,
                iso3,
                task_parent_id,
                model_time_point,
                census_time_point,
                model_spec,
            )

    if verbose:
        logger.info("Complete")


def check_time_points(census_weights: pd.DataFrame) -> None:
    test = census_weights.reset_index()
    test["match"] = test["model_time_point"] == test["census_time_point"]
    test = test.groupby(["iso3", "census_time_point"])["match"].any()

    if not test.all():
        raise ValueError(f"Missing raw_predictions for at least one time-point found in census data:\n{test}")


def check_complete(
    census_task: tuple[str, str, str],
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
):
    iso3, census_time_point, task_parent_id = census_task
    # A completed task always writes a raster at model_time_point == census_time_point
    # (weight 1 there). Use that as the marker: with multiple censuses per country a
    # census contributes to only a subset of model time points, so a fixed global last
    # time point would be missing for early censuses and they'd rerun forever.
    path = pm_data.raked_census_path(iso3, census_time_point, census_time_point, model_spec) / f"{task_parent_id}.tif"
    if not path.exists():
        return ()
    elif path.stat().st_size < 1:
        return ()
    else:
        return iso3, census_time_point, task_parent_id


# Per-task resources are predicted from each task's admin features, replacing the
# old xxl/xl/big/standard tiers (one workflow, individually sized tasks). Model
# calibrated on the 2026-07-06 full-run jobmon metadata (17,073 done tasks; see
# .claude/validate_census_rake/calibrate_per_task_resources.py):
#   MEMORY = margin * (floor
#            + census-cache term: the country census parquet is scanned through
#              the page cache each task (USA 11.7 GB -> ~26 GB floor)
#            + box term ~ bounds_area: template + scatter arrays span the full
#              bounding box (pop-0 maritime/arctic admins: huge bbox, little else)
#            + covered-land term ~ area for POPULATED admins: the covered_pixels
#              x n_tps working set (~0.22-0.24 GB per 1000 km^2, stable from 50k
#              to 522k km^2); population==0 admins are heavily water-masked, so
#              a small slope. NOTE a future *empty* fully-covered desert would be
#              under-predicted and lean on the retry.)
#   RUNTIME = margin * (floor + area term + perimeter term (convoluted CAN
#             boundaries: high runtime at low memory)).
# Coverage on the full run: 97.6% of tasks fit the first attempt; the 2.4% tail
# (worst obs/request ratio 1.24) is fully covered by jobmon's default +50% retry
# bump -- margins-not-maxima beats zero-retry tiers: 75% of the tiers' reserved
# memory and 52% of their reserved runtime. SAU.8_1 (the 96 G kill, est ~115 GB)
# gets a 166 G first attempt.
MEMORY_FLOOR_GB = 3.5
MEMORY_PER_CENSUS_GB = 2.0
MEMORY_PER_KKM2_BOX = 0.012
MEMORY_PER_KKM2_POPULATED = 0.24
MEMORY_PER_KKM2_EMPTY = 0.04
MEMORY_MARGIN = 1.15
MEMORY_BOUNDS_GB = (8, 240)
RUNTIME_FLOOR_MIN = 6.0
RUNTIME_PER_KKM2 = 0.09
RUNTIME_PER_KKM_PERIM = 2.5
RUNTIME_MARGIN = 1.7
RUNTIME_BOUNDS_MIN = (10, 240)


def build_task_resources(
    pm_data: PopulationModelData,
    task_admins: gpd.GeoDataFrame,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Predicted memory/runtime per (iso3, census_time_point, task_parent_id)."""
    features = task_admins.reset_index()
    census_gb = {
        (iso3, ctp): pm_data.census_path(iso3, ctp.split("q")[0]).stat().st_size / 1e9
        for iso3, ctp in features[["iso3", "census_time_point"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    cache_gb = pd.Series(
        [census_gb[key] for key in zip(features["iso3"], features["census_time_point"], strict=True)],
        index=features.index,
    )
    area_kkm2 = features["area"] / 1e9
    bounds_kkm2 = features["bounds_area"] / 1e9
    perim_kkm = features["perimeter"] / 1e6
    populated = features["population_total"] > 0

    memory_gb = MEMORY_MARGIN * (
        MEMORY_FLOOR_GB
        + MEMORY_PER_CENSUS_GB * cache_gb
        + MEMORY_PER_KKM2_BOX * bounds_kkm2
        + np.where(populated, MEMORY_PER_KKM2_POPULATED, MEMORY_PER_KKM2_EMPTY) * area_kkm2
    )
    memory_gb = np.ceil(memory_gb.clip(*MEMORY_BOUNDS_GB)).astype(int)
    runtime_min = RUNTIME_MARGIN * (
        RUNTIME_FLOOR_MIN
        + RUNTIME_PER_KKM2 * area_kkm2
        + RUNTIME_PER_KKM_PERIM * perim_kkm
    )
    runtime_min = np.ceil(runtime_min.clip(*RUNTIME_BOUNDS_MIN)).astype(int)

    return {
        (iso3, ctp, tpid): {"memory": f"{mem}G", "runtime": f"{run}m"}
        for iso3, ctp, tpid, mem, run in zip(
            features["iso3"],
            features["census_time_point"],
            features["task_parent_id"],
            memory_gb,
            runtime_min,
            strict=True,
        )
    }


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_iso3()
@click.option("--task-parent-id", type=str, required=True)
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def census_rake_task(
    resolution: str,
    version: str,
    iso3: str,
    time_point: str,
    task_parent_id: str,
    output_dir: str | Path,
) -> None:
    census_rake_main(
        resolution=resolution,
        version=version,
        iso3=iso3,
        census_time_point=time_point,
        task_parent_id=task_parent_id,
        output_dir=output_dir,
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def census_rake(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = sorted(clio.convert_choice(time_point, prediction_time_points))

    task_admins, census_weights = utils.generate_census_inputs(
        pm_data,
        start_level=1,
        # Coarser than the old (1e10, 20_000): post-fix, per-task memory is driven
        # by the per-country census-cache floor + box arrays (~bounds_area), not
        # shape count -- so we consolidate into fewer, bigger, longer jobs and pay
        # the census-cache floor fewer times. Calibrate up further if there's headroom.
        bounds_area_threshold=2e11,
        area_no_pop_threshold=1e12,
        admins_threshold=100_000,
    )
    census_weights = census_weights.loc[:, time_points, :]
    check_time_points(census_weights)

    census_tasks = (
        task_admins.reset_index()
        .loc[:, ["iso3", "census_time_point", "task_parent_id"]]
        .drop_duplicates()
    )

    pm_data.save_census_raking_metadata(
        task_admins,
        census_weights,
        census_tasks,
        model_spec
    )
    possible_census_tasks = list(census_tasks.itertuples(index=False, name=None))

    _check_complete = functools.partial(
        check_complete,
        pm_data=pm_data,
        model_spec=model_spec,
    )
    complete_census_tasks = parallel.run_parallel(
        _check_complete,
        possible_census_tasks,
        num_cores=10,
    )
    complete_census_tasks = [i for i in complete_census_tasks if len(i) > 0]

    task_resources_by_task = build_task_resources(pm_data, task_admins)

    complete = set(complete_census_tasks)
    census_tasks = [task for task in possible_census_tasks if task not in complete]
    if 0 < DOWNSAMPLE_ADMINS < len(census_tasks):
        census_tasks = pd.Series(census_tasks).sample(DOWNSAMPLE_ADMINS).sort_index().tolist()

    if len(census_tasks) > 0:
        print(
            f"Building raking factors for {len(census_tasks):,} census time-admins "
            f"(out of a possible {len(possible_census_tasks):,})."
        )
        jobmon.run_parallel(
            runner="pmtask postprocess",
            task_name="census_rake",
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": f"{MEMORY_BOUNDS_GB[0]}G",
                "runtime": f"{RUNTIME_BOUNDS_MIN[0]}m",
                "project": "proj_rapidresponse",
            },
            flat_node_args=(("iso3", "time-point", "task-parent-id"), census_tasks),
            per_task_resources=lambda args: task_resources_by_task[tuple(args)],
            task_args={
                "resolution": resolution,
                "version": version,
                "output-dir": output_dir,
            },
            max_attempts=2,
            log_root=pm_data.log_dir("postprocess_census_rake"),
            concurrency_limit=1_000,
        )
