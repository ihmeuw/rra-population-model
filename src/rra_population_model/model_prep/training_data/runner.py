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


# Tasks are sized from their tile neighbourhood - the tiles
# `get_training_metadata` loads - which is what sets a task's cost. A tile's own
# size tells you nothing; neighbourhoods run 1 to 197 and that range is the whole
# 20 GB to 240 GB spread.
#
# Calibrated on the 2026-09-04 run (jobmon model_prep_training_data/
# 2026_09_04_13_13_18; 37,722 attempts over 34,230 tiles) against the *smallest
# request each tile was observed to survive*. That target matters: an earlier fit
# to MaxRSS was discarded because MaxRSS measures what a task was given, not what
# it needed - on census raking, 9,308 tasks run at two different requests showed
# dRSS/dREQ = 0.353, page cache expanding into whatever is offered. Fitting
# reservations to RSS converges downward until tasks die; that fit scored 67.7%
# coverage against the 98.2% of the constants it meant to improve on.
#
# Memory is the binding resource, not runtime: OOM kills happened at 51-61% of
# their time limit, and every one of the 111 tasks that reached 240 GB completed.
# The bands carry runtime along for free - the 240 timeouts in the top band were
# large-neighbourhood tiles holding the 10m base while needing 13m median.
#
#   neighbourhood   tiles    memory  runtime   fit on the first attempt
#   <= 20          25,779      20 G     10 m   99.9%
#   21-40           2,095      40 G     20 m   99.6%
#   41-80           1,410      80 G     30 m   99.9%
#   > 80              570     240 G     90 m   94.4% (see below)
#
# The top band's figure is pessimistic: it counts 32 tiles that never succeeded,
# but those were never offered 240 GB - they exhausted their attempts at 80 GB.
# Preallocation is what fixes them, by starting them where they need to be
# instead of climbing to it.
BAND_RESOURCES: list[tuple[int, int, int]] = [
    # (max neighbourhood size, memory GB, runtime minutes)
    (20, 20, 10),
    (40, 40, 20),
    (80, 80, 30),
    (10_000, 240, 90),
]

# The banding above, resolved per tile and cached beside the module. Building it
# needs the tile neighbourhoods: ~6 minutes, and it loads the full USA max-level
# census (8.1M shapes) on the submitting host. Reading it costs milliseconds.
#
# Built on first use rather than shipped as package data, so there is nothing to
# declare in pyproject.toml and no way for an install to arrive without it. It is
# a cache, not source: delete it to force a rebuild after the modeling frame, the
# census vintages, or the denominator list change.
TASK_RESOURCE_TABLE = Path(__file__).parent / "task_resources.parquet"

# What a tile absent from the table gets. A miss means the table predates the
# current frame - expected rather than exceptional - but a miss sized as a
# typical tile is how the 2026-09-04 run lost 32 of them. Unknown means
# unmeasured, so it takes the top band.
FALLBACK_MEMORY_GB, FALLBACK_RUNTIME_MIN = BAND_RESOURCES[-1][1], BAND_RESOURCES[-1][2]


def build_task_resources(
    to_run: list[tuple[str, str, str]],
    queue: str,
    resolution: str,
    pm_data: PopulationModelData,
) -> dict[str, dict[str, str]]:
    """Memory and runtime per tile key, from the cached table.

    Builds and caches the table if it is not there yet.
    """
    if not TASK_RESOURCE_TABLE.exists():
        print(
            f"{TASK_RESOURCE_TABLE.name} not found; building it. This takes a few "
            f"minutes and is cached for subsequent runs."
        )
        table = utils.build_task_resource_table(resolution, pm_data, BAND_RESOURCES)
        table.to_parquet(TASK_RESOURCE_TABLE, index=False)
        print(f"  wrote {TASK_RESOURCE_TABLE}")

    table = pd.read_parquet(TASK_RESOURCE_TABLE).set_index("tile_key")
    tile_keys = [tile_key for tile_key, _, _ in to_run]
    missing = sorted(set(tile_keys) - set(table.index))
    if missing:
        print(
            f"WARNING: {len(missing):,} of {len(tile_keys):,} tiles are absent from "
            f"{TASK_RESOURCE_TABLE.name} and fall back to "
            f"{FALLBACK_MEMORY_GB}G/{FALLBACK_RUNTIME_MIN}m. Delete the file to "
            f"rebuild if this is more than a handful. First few: {missing[:5]}"
        )

    resources = {}
    for tile_key in tile_keys:
        if tile_key in table.index:
            row = table.loc[tile_key]
            memory, runtime = int(row["memory_gb"]), int(row["runtime_min"])
        else:
            memory, runtime = FALLBACK_MEMORY_GB, FALLBACK_RUNTIME_MIN
        resources[tile_key] = {
            "queue": queue,
            "cores": 1,
            "memory": f"{memory}G",
            "runtime": f"{runtime}m",
            "project": "proj_rapidresponse",
        }
    return resources


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

    task_resources_by_tile = build_task_resources(to_run, queue, resolution, pm_data)

    print(f"Building data for {len(to_run)} tiles.")
    status = jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="training_data",
        flat_node_args=(("tile-key", "time-point", "iso3-time-point-list"), to_run),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        # Floor only. Every task is sized individually below; this is what one
        # would get with no prediction, so it is the smallest band rather than a
        # guess at the typical task.
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": BAND_RESOURCES[0][1],
            "runtime": BAND_RESOURCES[0][2],
            "project": "proj_rapidresponse",
        },
        # args = (tile-key, time-point, iso3-time-point-list). A tile appears once
        # per time point and its neighbourhood does not depend on the time point,
        # so the tile key alone keys the prediction.
        per_task_resources=lambda args: task_resources_by_tile[args[0]],
        max_attempts=5,
        # A fraction, not an iterator. Jobmon applies a numeric scaler as
        # `ceil(value * (1 + factor))` per task, which is stateless; an Iterator
        # is a single shared object consumed with `next()` across the workflow,
        # and on StopIteration it silently reuses the previous value instead of
        # escalating. The 2026-09-04 run lost 32 tiles that stopped climbing at
        # 80 GB after three attempts while other tiles were still reaching 240 GB,
        # which is the failure that shape of config invites.
        #
        # This is now a backstop for a mis-banded tile, not the sizing mechanism:
        # +100% per attempt takes the top band 240 -> 480 GB if it is ever needed.
        resource_scales={"memory": 1.0, "runtime": 1.0},
        log_root=pm_data.log_dir("model_prep_training_data"),
    )

    if status != "D":
        msg = f"Workflow failed with status {status}."
        raise RuntimeError(msg)

    print("Building summary datasets.")
    people_per_structure = utils.build_summary_people_per_structure(pm_data, resolution)
    pm_data.save_summary_people_per_structure(people_per_structure, resolution)