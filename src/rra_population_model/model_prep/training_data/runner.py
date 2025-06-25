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
    iso3_year_list: str,
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

    iso3_year_list = [i.split('|') for i in iso3_year_list.split(",")]
    iso3_year_dict = (
        pd.DataFrame(
            [iso3_time_point for iso3_time_point in iso3_year_list],
            columns=['iso3', 'time_point']
        )
        .groupby(['time_point'])['iso3'].apply(list)
        .to_dict()
    )
    model_gdfs = []
    for data_time_point, iso3_list in iso3_year_dict.items():
        print(f"{data_time_point} - Finding intersecting admin units ({','.join(iso3_list)})")
        admins = utils.get_intersecting_admins(
            tile_meta=tile_meta,
            iso3_list=iso3_list,
            time_points=[data_time_point] * len(iso3_list),
            pm_data=pm_data,
        )
        if admins.empty:
            continue

        print(f"{data_time_point} - Getting training metadata")
        training_meta = get_training_metadata(
            tile_meta=tile_meta,
            model_frame=model_frame,
            resolution=resolution,
            time_point=data_time_point,
            intersecting_admins=admins,
            pm_data=pm_data,
        )

        print(f"{data_time_point} - Loading model gdfs")
        for n_tile_meta in training_meta.tile_neighborhood:
            print(n_tile_meta.key)
            n_tile_gdf = utils.get_tile_feature_gdf(
                n_tile_meta,
                training_meta,
                pm_data,
            )
            if not n_tile_gdf.empty:
                model_gdfs.append(n_tile_gdf)

    if not model_gdfs:
        print("No intersecting admin units found. Likely open ocean.")
        return

    print("Processing model gdf")
    model_gdf = pd.concat(model_gdfs, ignore_index=True)
    model_gdf['time_point'] = time_point
    admins = utils.get_intersecting_admins(
        tile_meta=tile_meta,
        iso3_list=[i[0] for i in iso3_year_list],
        time_points=[i[1] for i in iso3_year_list],
        pm_data=pm_data,
    )
    training_meta = get_training_metadata(
        tile_meta=tile_meta,
        model_frame=model_frame,
        resolution=resolution,
        time_point=time_point,
        intersecting_admins=admins,
        pm_data=pm_data,
    )
    model_gdf = utils.process_model_gdf(model_gdf, training_meta)
    admin_gdf = utils.filter_to_admin_gdf(model_gdf, training_meta)
    tile_gdf = model_gdf[model_gdf["tile_key"] == tile_key]

    if purpose == 'training':
        print("Calculating pixel area weights")
        pixel_area_weight = (
            tile_gdf.groupby(["admin_id", "pixel_id"])[["admin_area_weight"]]
            .first()
            .reset_index()
        )

    print("rasterizing features")
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
        out_measures = ["occupancy_rate"]  # , "density", "residential_volume"
    training_rasters = [
        f"{m}_{d}"
        for m, d in itertools.product(out_measures, training_meta.denominators)
    ] + ["multi_tile"]
    tile_rasters = {}
    for raster_name in training_rasters:
        raster = utils.raster_from_pixel_feature(tile_gdf, raster_name, raster_template)
        tile_rasters[raster_name] = raster

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


@click.command()
@click.option("--iso3-year-list", type=str, required=True)
@clio.with_resolution()
@clio.with_tile_key()
@clio.with_time_point()
@clio.with_purpose()
@clio.with_output_directory(pmc.MODEL_ROOT)
def training_data_task(
    resolution: str,
    iso3_year_list: str,
    tile_key: str,
    time_point: str,
    purpose: str,
    output_dir: str,
) -> None:
    """Build the response for a given tile and time point."""
    training_data_main(
        resolution,
        iso3_year_list,
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
    print(f"Building test/train data for {len(to_run)} tiles.")

    status = jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="training_data",
        flat_node_args=(("tile-key", "time-point", "iso3-year-list"), to_run),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
            "purpose": purpose,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=5,
        log_root=pm_data.log_dir("model_prep_training_data"),
    )

    if status != "D":
        msg = f"Workflow failed with status {status}."
        raise RuntimeError(msg)

    if purpose == 'training':
        print("Building summary datasets.")
        people_per_structure = utils.build_summary_people_per_structure(pm_data, resolution)
        pm_data.save_summary_people_per_structure(people_per_structure, resolution)
    elif purpose != 'inference':
        raise ValueError(f'Unexpected data purpose: {purpose}')
