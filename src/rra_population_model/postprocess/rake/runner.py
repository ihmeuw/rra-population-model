import click
import numpy as np
import rasterra as rt
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon
import geopandas as gpd
import shapely

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import (
    block_census_tasks,
    get_prediction_time_point,
    load_block_census_layer,
    repair_invalid_geometries,
)

STAGES = ["final", "gbd"]


def build_raking_factor_raster(
    raking_data: gpd.GeoDataFrame,
    like: rt.RasterArray,
) -> rt.RasterArray:
    """Rasterize per-admin raking factors onto ``like``'s grid (NaN outside)."""
    raking_factor_data = np.nan * np.ones_like(like)
    for geom, rf in raking_data[["geometry", "raking_factor"]].itertuples(index=False):
        shape_mask, *_ = raster_geometry_mask(
            data_transform=like.transform,
            data_width=like.shape[1],
            data_height=like.shape[0],
            shapes=[geom],
            invert=True,
        )
        raking_factor_data[shape_mask] = rf
    return rt.RasterArray(
        raking_factor_data,
        transform=like.transform,
        crs=like.crs,
        no_data_value=np.nan,
    )


def rake_main(
    resolution: str,
    version: str,
    stage: str,
    census: bool,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    """Write one raked block raster, computed entirely from immutable inputs.

    stage "final" (the pipeline product, written once to raked_predictions/):
        (raw x initial raking factor) -> splice census layer -> x final raking
        factor. With ``census=False`` (GBD-only model versions, e.g. validation
        runs) the splice and the final factor are skipped and raw x initial IS
        the version's final product.
    stage "gbd" (on-demand product, written to gbd_raked_predictions/):
        raw x initial raking factor -- the GBD-raked surface without census
        information, used by the pseudo-OOS validation against censuses.

    No stage reads anything another rake task writes, so retries are idempotent
    and reruns are safe.
    """
    pm_data = PopulationModelData(output_dir)

    print("Loading metadata")
    model_spec = pm_data.load_model_specification(resolution, version)
    prediction_time_point = get_prediction_time_point(
        pm_data, resolution, version, time_point
    )
    print("Loading raw prediction")
    unraked_data = pm_data.load_raw_prediction(
        block_key, prediction_time_point, model_spec
    )

    print("Loading initial raking factors")
    raking_data = pm_data.load_raking_factors(
        time_point,
        model_spec,
        stage="initial",
        filters=[("block_key", "==", block_key)],
    )

    print("Raking")
    if raking_data.empty:
        raked = rt.RasterArray(
            np.nan * np.ones_like(unraked_data),
            transform=unraked_data.transform,
            crs=unraked_data.crs,
            no_data_value=np.nan,
        )
    else:
        raked = unraked_data * build_raking_factor_raster(raking_data, unraked_data)

        if stage == "final" and census:
            print("Splicing census layer")
            model_frame = pm_data.load_modeling_frame(resolution)
            model_frame = model_frame.loc[model_frame["block_key"] == block_key]
            block_geometry = model_frame.union_all()

            task_admins, census_weights = pm_data.load_census_raking_inputs(model_spec)
            task_admins = repair_invalid_geometries(task_admins)
            census_layer = load_block_census_layer(
                pm_data,
                model_spec,
                block_geometry,
                prediction_time_point,
                block_census_tasks(task_admins, block_geometry),
                census_weights,
            )
            if census_layer is not None:
                raked = rt.merge([census_layer, raked], method="first")

            print("Applying final raking factors")
            final_raking_data = pm_data.load_raking_factors(
                time_point,
                model_spec,
                stage="final",
                filters=[("block_key", "==", block_key)],
            )
            if set(final_raking_data["location_id"]) != set(raking_data["location_id"]):
                raise ValueError(
                    "Initial and final raking factors disagree on this block's "
                    "admins; both stages must be built from the same raking shapes."
                )
            raked = raked * build_raking_factor_raster(final_raking_data, raked)

    print("Saving")
    if stage == "final":
        pm_data.save_raked_prediction(raked, block_key, time_point, model_spec)
    else:
        pm_data.save_gbd_raked_prediction(raked, block_key, time_point, model_spec)


def create_bounds_polygon(raster_data: rt.RasterArray) -> shapely.Polygon:
    bounds = raster_data.bounds
    bounds = (
        bounds[0], bounds[2],
        bounds[1], bounds[3],
    )
    bounds = (
        gpd.GeoSeries(
            [shapely.box(*bounds)],
            crs=raster_data.crs
        )
        .explode(index_parts=True)
        .union_all()
    )

    return bounds


@click.command()
@clio.with_resolution()
@clio.with_version()
@click.option("--stage", type=click.Choice(STAGES), required=True)
@click.option("--census", type=bool, default=True, show_default=True)
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_task(
    resolution: str,
    version: str,
    stage: str,
    census: bool,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_main(resolution, version, stage, census, block_key, time_point, output_dir)


@click.command()
@clio.with_resolution(allow_all=False)
@clio.with_version()
@click.option("--stage", type=click.Choice(STAGES), required=True)
@click.option("--census", type=bool, default=True, show_default=True)
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake(
    resolution: str,
    version: str,
    stage: str,
    census: bool,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    if stage == "gbd" and not census:
        raise ValueError("--census False only applies to --stage final.")

    # Fail fast on missing raking factors rather than mid-workflow.
    rf_time_points = set(
        pm_data.list_raking_factor_time_points(resolution, version, stage="initial")
    )
    splicing = stage == "final" and census
    if splicing:
        rf_time_points &= set(
            pm_data.list_raking_factor_time_points(resolution, version, stage="final")
        )
    time_points = clio.convert_choice(time_point, sorted(rf_time_points))

    model_frame = pm_data.load_modeling_frame(resolution)
    block_keys = model_frame.block_key.unique().tolist()

    if resolution == "40":
        if splicing:
            task_resources = {
                "queue": queue,
                "cores": 1,
                "memory": "9G",
                "runtime": "9m",
                "project": "proj_rapidresponse",
            }
        else:
            task_resources = {
                "queue": queue,
                "cores": 1,
                "memory": "4G",
                "runtime": "4m",
                "project": "proj_rapidresponse",
            }
    elif resolution == "100":
        task_resources = {
            "queue": queue,
            "cores": 1,
            "memory": "6G",
            "runtime": "6m",
            "project": "proj_rapidresponse",
        }

    print(f"Raking {len(block_keys) * len(time_points)} blocks")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake",
        task_resources=task_resources,
        node_args={
            "block-key": block_keys,
            "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
            "stage": stage,
            "census": census,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("postprocess_rake"),
    )
