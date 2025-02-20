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
    iso3_list: str,
    tile_key: str,
    time_point: str,
    model_root: str | Path,
) -> None:
    """Build the training data for the MEX model for a single tile."""
    print("Loading metadata")
    pm_data = PopulationModelData(model_root)
    model_frame = pm_data.load_modeling_frame(resolution)
    tile_meta = TileMetadata.from_model_frame(model_frame, tile_key)
    print("Finding intersecting admin units")
    admins = utils.get_intersecting_admins(
        tile_meta=tile_meta,
        iso3_list=iso3_list,
        time_point=time_point,
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

    print("Loading model gdfs")
    model_gdfs = []
    for n_tile_meta in training_meta.tile_neighborhood:
        print(n_tile_meta.key)
        n_tile_gdf = utils.get_tile_feature_gdf(
            n_tile_meta,
            training_meta,
            pm_data,
        )
        if not n_tile_gdf.empty:
            model_gdfs.append(n_tile_gdf)

    print("Processing model gdf")
    model_gdf = pd.concat(model_gdfs, ignore_index=True)

    model_gdf = utils.process_model_gdf(model_gdf, training_meta)
    admin_gdf = utils.filter_to_admin_gdf(model_gdf, training_meta)
    tile_gdf = model_gdf[model_gdf["tile_key"] == tile_key]

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

    out_measures = ["population", "occupancy_rate", "log_occupancy_rate"]
    training_rasters = [
        f"{m}_{d}"
        for m, d in itertools.product(out_measures, training_meta.denominators)
    ] + ["multi_tile"]
    tile_rasters = {}
    for raster_name in training_rasters:
        raster = utils.raster_from_pixel_feature(tile_gdf, raster_name, raster_template)
        tile_rasters[raster_name] = raster

    print("Saving")
    pm_data.save_tile_training_data(
        resolution,
        tile_key,
        admin_gdf,
        pixel_area_weight,
        tile_rasters,
    )


@click.command()  # type: ignore[arg-type]
@click.option("--iso3-list", type=str, required=True)
@clio.with_resolution()
@clio.with_tile_key()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def training_data_task(
    resolution: str,
    iso3_list: str,
    tile_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    """Build the response for a given tile and time point."""
    training_data_main(
        resolution,
        iso3_list,
        tile_key,
        time_point,
        output_dir,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def training_data(
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Build the training data for the MEX model."""
    pm_data = PopulationModelData(output_dir)

    print("Building arg list")
    to_run = utils.build_arg_list(resolution, pm_data)
    print(f"Building test/train data for {len(to_run)} tiles.")

    status = jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="training_data",
        flat_node_args=(("tile-key", "time-point", "iso3-list"), to_run),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        max_attempts=5,
    )

    if status != "D":
        msg = f"Workflow failed with status {status}."
        raise RuntimeError(msg)

    print("Building summary datasets.")
    people_per_structure, pixel_area_weight = utils.build_summary_data(
        pm_data, resolution
    )

    pm_data.save_summary_training_data(
        resolution,
        people_per_structure,
        pixel_area_weight,
    )
