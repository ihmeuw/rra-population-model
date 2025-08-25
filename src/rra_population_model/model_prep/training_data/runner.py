import itertools
from pathlib import Path

import click
import pandas as pd
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model_prep.training_data import utils
from rra_population_model.model_prep.training_data.metadata import (
    TileMetadata,
    get_training_metadata,
)


def training_data_main(
    resolution: str,
    iso3_time_point_list: str,
    tile_key: str,
    time_point: str,
    purpose: str,
    model_root: str | Path,
) -> None:
    """Build the training data for the model for a single tile."""
    print("Loading metadata")
    pm_data = PopulationModelData(model_root)
    model_frame = pm_data.load_modeling_frame(resolution)
    tile_meta = TileMetadata.from_model_frame(model_frame, tile_key)

    iso3_time_point_list = [i.split(':') for i in iso3_time_point_list.split(",")]
    print("Finding intersecting admin units")
    admins = utils.get_intersecting_admins(
        tile_meta=tile_meta,
        iso3_time_point_list=iso3_time_point_list,
        pm_data=pm_data,
    )
    if admins.empty:
        print("No intersecting admin units found. Likely open ocean.")
        return

    print("Getting training metadata")
    training_meta = get_training_metadata(
        tile_meta=tile_meta,
        model_frame=model_frame,
        resolution=resolution,
        time_point=time_point,
        intersecting_admins=admins,
        pm_data=pm_data,
    )

    model_gdfs = []
    data_time_point_list = list(set([i[1] for i in iso3_time_point_list] + [time_point]))
    for data_time_point in data_time_point_list:
        print(f"Loading model gdfs -- {data_time_point}")
        time_point_model_gdfs = []
        for n_tile_meta in training_meta.tile_neighborhood:
            print(n_tile_meta.key)
            n_tile_gdf = utils.get_tile_feature_gdf(
                tile_meta=n_tile_meta,
                training_meta=training_meta,
                pm_data=pm_data,
                time_point=data_time_point,
            )
            if not n_tile_gdf.empty:
                time_point_model_gdfs.append(n_tile_gdf)

        print(f"Processing model gdf-- {data_time_point}")
        time_point_model_gdf = pd.concat(
            time_point_model_gdfs, ignore_index=True
        )
        time_point_model_gdf = utils.process_model_gdf(
            time_point_model_gdf, training_meta
        )
        model_gdfs.append(time_point_model_gdf)
    model_gdf = pd.concat(model_gdfs, ignore_index=True)
    tile_gdf = model_gdf[model_gdf["tile_key"] == tile_key]

    if purpose == 'training':
        model_gdf = model_gdf.loc[model_gdf['time_point'] == time_point]
        admin_gdf = utils.filter_to_admin_gdf(model_gdf, training_meta)

        print("Calculating pixel area weights")
        pixel_area_weight = (
            tile_gdf.groupby(["admin_id", "pixel_id"])[["admin_area_weight"]]
            .first()
            .reset_index()
        )
    elif purpose == 'inference':
        tile_gdfs = []
        for data_time_point in data_time_point_list:
            time_point_tile_gdfs = []
            for denominator in training_meta.denominators:
                pixel_occupancy_rate = (
                    tile_gdf
                    .loc[(tile_gdf['census_time_point'] == data_time_point) & (tile_gdf['time_point'] == data_time_point)]
                    .set_index(['admin_id', 'pixel_id'])
                    .loc[:, f'pixel_occupancy_rate_{denominator}']
                )
                pixel_built = (
                    tile_gdf
                    .loc[(tile_gdf['census_time_point'] == data_time_point) & (tile_gdf['time_point'] == time_point)]
                    .set_index(['admin_id', 'pixel_id'])
                    .loc[:, f'pixel_built_{denominator}']
                )
                time_point_tile_gdfs.append(
                    (pixel_occupancy_rate * pixel_built)
                    .rename(f'pixel_population_{denominator}')
                )
            tile_gdfs.append(
                pd.concat(time_point_tile_gdfs, axis=1).reset_index()
            )
        multi_tile = (
            tile_gdf
            .loc[tile_gdf['time_point'] == time_point]
            .loc[:, ['admin_id', 'pixel_id', 'pixel_multi_tile']]
        )
        area_weight = (
            tile_gdf
            .loc[tile_gdf['time_point'] == time_point]
            .loc[:, ['admin_id', 'pixel_id', 'pixel_area_weight']]
        )
        tile_gdf = pd.concat(tile_gdfs, ignore_index=True).merge(multi_tile).merge(area_weight)
    else:
        raise ValueError(f'Invalid purpose: {purpose}')

    print("Rasterizing features")
    raster_template = pm_data.load_feature(
        resolution=resolution,
        block_key=tile_meta.block_key,
        feature_name=training_meta.denominators[0],
        time_point=time_point,
        subset_bounds=tile_meta.polygon,
    )

    if purpose == 'training':
        out_measures = ["population", "occupancy_rate", "log_occupancy_rate"]
    elif purpose == 'inference':
        out_measures = ["population"]
    training_rasters = [
        f"{m}_{d}"
        for m, d in itertools.product(out_measures, training_meta.denominators)
    ] + ["multi_tile"]
    if purpose == "inference":
        training_rasters.append("area_weight")
    tile_rasters = {}
    for raster_name in training_rasters:
        raster = utils.raster_from_pixel_feature(tile_gdf, raster_name, raster_template)
        tile_rasters[raster_name] = raster

    # import numpy as np
    # if purpose == "inference":
    #     print(
    #         f"Raster total: {np.nansum(tile_rasters['population_microsoft_v7_1_residential_volume'])}"
    #     )
    #     model_gdf = model_gdf.loc[model_gdf['time_point'] == time_point]
    #     admin_gdf = utils.filter_to_admin_gdf(model_gdf, training_meta)
    #     print(
    #         f"Admin total: {admin_gdf['admin_population'].sum()}"
    #     )

    print("Saving")
    if purpose == 'training':
        pm_data.save_tile_training_data(
            resolution,
            tile_key,
            admin_gdf,
            pixel_area_weight,
            tile_rasters,
        )
    elif purpose == 'inference':
        pm_data.save_tile_inference_data(
            resolution,
            time_point,
            tile_key,
            tile_rasters,
        )
    else:
        raise ValueError(f'Unexpected data purpose: {purpose}')


@click.command()
@click.option("--iso3-time-point-list", type=str, required=True)
@clio.with_resolution()
@clio.with_tile_key()
@clio.with_time_point()
@clio.with_purpose()
@clio.with_output_directory(pmc.MODEL_ROOT)
def training_data_task(
    resolution: str,
    iso3_time_point_list: str,
    tile_key: str,
    time_point: str,
    purpose: str,
    output_dir: str,
) -> None:
    """Build the response for a given tile and time point."""
    training_data_main(
        resolution,
        iso3_time_point_list,
        tile_key,
        time_point,
        purpose,
        output_dir,
    )


@click.command()
@clio.with_resolution()
@clio.with_purpose()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def training_data(
    resolution: str,
    purpose: str,
    output_dir: str,
    queue: str,
) -> None:
    """Build the training data for the model."""
    pm_data = PopulationModelData(output_dir)

    print("Building arg list")
    to_run = utils.build_arg_list(resolution, pm_data, purpose)
    to_run = [i for i in to_run if i[1].startswith('202')]

    time_points = sorted(list(set([i[1] for i in to_run])))
    years = sorted(list(set([time_point.split('q')[0] for time_point in time_points])))
    for year in years:
        to_run_year = [i for i in to_run if i[1].startswith(year)]
        # year = '2023q1-2024q2'
        # to_run_year = [i for i in to_run if i[1].startswith('2023') or i[1].startswith('2024')]
        print(f"Building {purpose} data for {len(to_run_year)} tiles for {year}.")
        status = jobmon.run_parallel(
            runner="pmtask model_prep",
            task_name="training_data",
            flat_node_args=(("tile-key", "time-point", "iso3-time-point-list"), to_run_year),
            task_args={
                "output-dir": output_dir,
                "resolution": resolution,
                "purpose": purpose,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "10G",
                "runtime": "5m",
                "project": "proj_rapidresponse",
            },
            max_attempts=5,
            resource_scales={
                "memory":  iter([20     , 40     , 80     , 180    ]),  # G
                "runtime": iter([10 * 60, 20 * 60, 30 * 60, 80 * 60]),  # seconds
            },
            log_root=pm_data.log_dir("model_prep_training_data"),
        )

        if status != "D":
            msg = f"Workflow failed with status {status}."
            raise RuntimeError(msg)
        print("############################################################")

    if purpose == 'training':
        print("Building summary datasets.")
        people_per_structure = utils.build_summary_people_per_structure(pm_data, resolution)
        pm_data.save_summary_people_per_structure(people_per_structure, resolution)
    elif purpose != 'inference':
        raise ValueError(f'Unexpected data purpose: {purpose}')
