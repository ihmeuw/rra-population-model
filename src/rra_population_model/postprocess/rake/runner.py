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
from rra_population_model.postprocess.utils import get_prediction_time_point


def rake_main(
    resolution: str,
    version: str,
    input_data: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Loading metadata")
    model_spec = pm_data.load_model_specification(resolution, version)
    prediction_time_point = get_prediction_time_point(
        pm_data, resolution, version, time_point
    )
    print("Loading unraked prediction")
    if input_data in ["raw", "raw_skip"]:
        unraked_data = pm_data.load_raw_prediction(
            block_key, prediction_time_point, model_spec
        )
    elif input_data == "raked":
        unraked_data = pm_data.load_raked_prediction(
            block_key, prediction_time_point, model_spec
        )

    print("Loading raking factors")
    raking_data = pm_data.load_raking_factors(
        time_point,
        model_spec,
        filters=[("block_key", "==", block_key)],
    )

    print("Raking")
    if raking_data.empty:
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

        if input_data == "raw":
            print("Loading raked_census data")
            model_frame = pm_data.load_modeling_frame(resolution)
            model_frame = model_frame.loc[model_frame["block_key"] == block_key]
            block_geometry = model_frame.union_all()

            task_admins, census_weights = pm_data.load_census_raking_inputs(model_spec)
            task_admins = task_admins.loc[task_admins.intersects(block_geometry)]
            task_admins = list(
                task_admins
                .reset_index()
                .loc[:, ["iso3", "task_parent_id", "census_time_point"]]
                .itertuples(index=False, name=None)
            )
            census_population = []
            for iso3, shape_id, census_time_point in task_admins:
                raked_census = pm_data.load_raked_census(
                    iso3,
                    shape_id,
                    prediction_time_point,
                    census_time_point,
                    model_spec,
                )
                if np.isnan(raked_census.to_numpy()).all():
                    # Admins with no predicted pixels write a minimal all-nodata
                    # raster; drop it here (it contributes nothing and would only
                    # inflate the merge extent). A *missing* tif raises above,
                    # surfacing a failed job instead of silently dropping it.
                    continue
                census_population.append(
                    raked_census
                    *
                    census_weights.loc[iso3, prediction_time_point, census_time_point].item()
                )
            if census_population:
                census_population = rt.merge(census_population, method="sum")
                census_population = census_population.clip(block_geometry).mask(block_geometry)
                raked = rt.merge([census_population, raked], method="first")

        elif input_data not in ["raked", "raw_skip"]:
            raise ValueError(f"Invalid `input_data` type: {input_data}")

    print("Saving raked prediction")
    pm_data.save_raked_prediction(raked, block_key, time_point, model_spec)


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
@click.option("--input-data", type=str, required=True)
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_task(
    resolution: str,
    version: str,
    input_data: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_main(resolution, version, input_data, block_key, time_point, output_dir)


@click.command()
@clio.with_resolution(allow_all=False)
@clio.with_version()
@click.option("--input-data", type=str, required=True)
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake(
    resolution: str,
    version: str,
    input_data: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    if input_data in ["raw", "raw_skip"]:
        if len(list(pm_data.raked_predictions_root(resolution, version).iterdir())) > 0:
            raise ValueError(f"Raked predictions already exist, cannot run with `input_data` set to `raw`.")
    elif input_data not in ["raked"]:
        raise ValueError(f"Invalid `input_data` type: {input_data}")

    rf_time_points = pm_data.list_raking_factor_time_points(resolution, version)
    time_points = clio.convert_choice(time_point, rf_time_points)

    model_frame = pm_data.load_modeling_frame(resolution)
    block_keys = model_frame.block_key.unique().tolist()

    # versions = [f"2025_11_08.0{(i + 1):02d}" for i in range(60)]
    # time_points = ["2020q1", "2020q2"]

    if resolution == "40":
        task_resources = {
            "queue": queue,
            "cores": 1,
            "memory": "5G",
            "runtime": "5m",
            "project": "proj_rapidresponse",
        }
    elif resolution == "100":
        task_resources = {
            "queue": queue,
            "cores": 1,
            "memory": "4G",
            "runtime": "3m",
            "project": "proj_rapidresponse",
        }

    print(f"Raking {len(block_keys) * len(time_points)} blocks")
    # for time_point in time_points:
    # print("##############################################################")
    # print(f"Raking {len(block_keys) * len(versions)} blocks for {time_point}")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake",
        task_resources=task_resources,
        node_args={
            # "version": versions,
            "block-key": block_keys,
            "time-point": time_points,
        },
        task_args={
            "version": version,
            # "time-point": time_point,
            "resolution": resolution,
            "output-dir": output_dir,
            "input-data": input_data,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("postprocess_rake"),
    )
