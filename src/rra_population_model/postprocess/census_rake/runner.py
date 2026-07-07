from pathlib import Path
from loguru import logger
import functools

import click
from rra_tools import jobmon, parallel

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


def build_workflows(
    task_admins: gpd.GeoDataFrame, queue: str
) -> dict[str, dict[str, dict[str, str | int | dict] | pd.Series]]:
    task_admins = task_admins.reset_index("task_parent_level", drop=True)
    common = {"queue": queue, "cores": 1, "project": "proj_rapidresponse"}
    iso3 = task_admins.index.get_level_values("iso3")

    # Post-overlay-speedup, MEMORY is the binding constraint and it scales with
    # covered pixels ~ area (process_census_data builds a covered_pixels x n_tps
    # prediction table), independent of population -- a near-empty huge land admin
    # still processes every pixel. Runtime is no longer the driver (heavy admins
    # finish in ~10-15 min after the integer-reindex + merge-free rake).
    #   * xl:  area > 5e10 m^2 (50k km^2) -> ~52 GB peak for the worst case (CAN,
    #          236k km^2, measured), so the 64 G first attempt succeeds. Also catches
    #          the one extreme-perimeter admin (>4000 km) for runtime headroom.
    #   * big: USA (11.7 GB census-cache floor -> ~26 GB) or a very large bounding
    #          box (pop=0 maritime singles).
    #   * standard: everything else (<= ~16 GB, minutes).
    is_xl = (task_admins["area"] > 5e10) | (task_admins["perimeter"] > 4e6)
    is_big = ((iso3 == "USA") | (task_admins["bounds_area"] > 5e11)) & ~is_xl

    workflows = {
        "xl": {
            "kwargs": {
                "task_resources": {**common, "memory": "64G", "runtime": "60m"},
                "max_attempts": 2,
                "resource_scales": {
                    "memory":  iter([96     ]),  # G retry insurance (peak ~52 G, so 64 G should hold)
                    "runtime": iter([90 * 60]),  # seconds
                },
            },
            "task_idx": task_admins.loc[is_xl].index,
        },
        "big": {
            "kwargs": {
                "task_resources": {**common, "memory": "32G", "runtime": "45m"},
                "max_attempts": 3,
                "resource_scales": {
                    "memory":  iter([48     , 64      ]),  # G
                    "runtime": iter([90 * 60, 150 * 60]),  # seconds
                },
            },
            "task_idx": task_admins.loc[is_big].index,
        },
        "standard": {
            "kwargs": {
                "task_resources": {**common, "memory": "18G", "runtime": "30m"},
                "max_attempts": 3,
                "resource_scales": {
                    "memory":  iter([30     , 45      ]),  # G
                    "runtime": iter([60 * 60, 120 * 60]),  # seconds
                },
            },
            "task_idx": task_admins.loc[~is_xl & ~is_big].index,
        },
    }
    return workflows


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
        start_level=2,
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

    workflows = build_workflows(task_admins, queue)

    for workflow_name, workflow in workflows.items():
        census_tasks = (
            workflow["task_idx"]
            .drop(complete_census_tasks, errors="ignore")
        )
        if 0 < DOWNSAMPLE_ADMINS < len(census_tasks):
            census_tasks = census_tasks.to_frame().sample(DOWNSAMPLE_ADMINS).sort_index().index
        census_tasks = census_tasks.tolist()

        if len(census_tasks) > 0:
            print(
                "\n"
                f"WORKFLOW: {workflow_name}\n"
                f"Building raking factors for {len(census_tasks):,} census time-admins (out of a possible {len(workflow["task_idx"]):,})."
            )
            jobmon.run_parallel(
                runner="pmtask postprocess",
                task_name="census_rake",
                task_resources=workflow["kwargs"]["task_resources"],
                flat_node_args=(("iso3", "time-point", "task-parent-id"), census_tasks),
                task_args={
                    "resolution": resolution,
                    "version": version,
                    "output-dir": output_dir,
                },
                max_attempts=workflow["kwargs"]["max_attempts"],
                resource_scales=workflow["kwargs"]["resource_scales"],
                log_root=pm_data.log_dir("postprocess_census_rake"),
                concurrency_limit=1_000,
            )
