import itertools
from pathlib import Path

import click
import numpy as np
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
    output_dir: str | Path,
) -> None:
    """Build the training data for the model for a single tile."""
    print("Loading metadata")
    pm_data = PopulationModelData(output_dir)
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

    model_gdf = model_gdf.loc[model_gdf['time_point'] == time_point]
    admin_gdf = utils.filter_to_admin_gdf(model_gdf, training_meta)

    print("Calculating pixel area weights")
    pixel_area_weight = (
        tile_gdf.groupby(["admin_id", "pixel_id"])[["admin_area_weight"]]
        .first()
        .reset_index()
    )

    print("Rasterizing features")
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
    pm_data.save_tile_training_data(
        resolution,
        tile_key,
        admin_gdf,
        pixel_area_weight,
        tile_rasters,
    )


@click.command()
@click.option("--iso3-time-point-list", type=str, required=True)
@clio.with_resolution()
@clio.with_tile_key()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def training_data_task(
    resolution: str,
    iso3_time_point_list: str,
    tile_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    """Build the response for a given tile and time point."""
    training_data_main(
        resolution,
        iso3_time_point_list,
        tile_key,
        time_point,
        output_dir,
    )


# Per-task sizing from the tile neighbourhood was built, validated, and is
# deliberately NOT wired in. `utils.build_neighbourhood_sizes` reproduces a
# task's neighbourhood exactly (100.00% on 1,910 tiles) and predicts peak RSS
# from it at R^2 0.942, so the geometry half is sound. The target was not: the
# relation (a 4.76 GB floor plus 0.5932 GB per neighbouring tile) was fitted to
# MaxRSS, and MaxRSS is not the memory a task needs. Measured on
# census raking, where 9,308 tasks ran at two materially different requests,
# dRSS/dREQ was 0.353 - the same task given 8 G peaks near 3 G and given 18 G
# peaks near 7 G, because page cache expands into whatever is offered. Fitting
# reservations to RSS is therefore a feedback loop that converges downward until
# tasks start dying, and on census raking a fit built exactly this way scored
# 67.7% coverage against the constants it was meant to improve on (98.2%).
#
# To turn it on, recalibrate against the smallest request each tile is observed
# to survive, gathered over several runs - not MaxRSS, and not one run. There is
# no such history yet because this scheme has never run.
@click.command()
@clio.with_resolution()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def training_data(
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Build the training data for the model."""
    pm_data = PopulationModelData(output_dir)

    print("Building arg list")
    to_run = utils.build_arg_list(resolution, pm_data)

    print(f"Building data for {len(to_run)} tiles.")
    status = jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="training_data",
        flat_node_args=(("tile-key", "time-point", "iso3-time-point-list"), to_run),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        # Raised from 10G/5m after the 2026-09-02 run (jobmon
        # model_prep_training_data/2026_09_02_11_14_36, 61,883 attempts).
        #
        # This rests on observed kills, not on a fit: 22,870 of 34,230 first
        # attempts (67%) died OUT_OF_MEMORY at 10G, so the base was demonstrably
        # below the typical task rather than merely under a modelled estimate.
        # 20G covers 93.3% first time (16G: 90.9%, 24G: 94.6% - 20 is the knee).
        # Runtime likewise: p90 4.0m and p99 11.2m against a 5m base produced
        # 1,781 TIMEOUTs.
        #
        # It buys cluster resources rather than wall clock - the retries ran
        # concurrently, so that run still finished in 1.40h - but it removes
        # ~1,000 wasted task-hours and ~20,000 GB-hours, 46% of everything the
        # run consumed.
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "20G",
            "runtime": "10m",
            "project": "proj_rapidresponse",
        },
        max_attempts=4,
        # The old first rung (20G/10m) is now the base, so the ladder starts
        # above it. The upper rungs still earn their place: measured p99 was
        # 59.4G against a 127.9G maximum.
        resource_scales={
            "memory":  iter([40     , 80     , 240    ]),  # G
            "runtime": iter([20 * 60, 30 * 60, 90 * 60]),  # seconds
        },
        log_root=pm_data.log_dir("model_prep_training_data"),
    )

    if status != "D":
        msg = f"Workflow failed with status {status}."
        raise RuntimeError(msg)

    print("Building summary datasets.")
    people_per_structure = utils.build_summary_people_per_structure(pm_data, resolution)
    pm_data.save_summary_people_per_structure(people_per_structure, resolution)