import sys
from pathlib import Path
from typing import List
from loguru import logger
import tqdm

import numpy as np
import rasterra as rt
from rasterio.windows import WindowError
import pandas as pd
import geopandas as gpd

from jobmon.client.tool import Tool
import uuid
import shutil
from rra_tools.shell_tools import mkdir

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

LSVID = 1285
SCRATCH_DIR = "2025-12-03-urban_rural_populations"


def execute_workflow(
    resolution: str,
    version: str,
    location_ids: List[int],
    radius: int,
):
    wf_uuid = uuid.uuid4()

    tool = Tool(name="urban_rural_admin")

    workflow = tool.create_workflow(
        name=f"urban_rural_admin_{wf_uuid}",
    )

    ## define templates
    task_template = tool.get_task_template(
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": "12G",
            "runtime": "4m",
            "project": "proj_rapidresponse",
        },
        template_name="urban_rural_admin",
        default_cluster_name="slurm",
        command_template=f"{shutil.which('python')}"
                         f" {Path(__file__)}"
                         " worker"
                         " {resolution}"
                         " {version}"
                         " {location_id}"
                         " {radius}",
        node_args=["location_id"],
        task_args=["resolution", "version", "radius"],
        op_args=[],
    )

    ## compile tasks
    tasks = []
    for location_id in location_ids:
        tasks.append(
            task_template.create_task(
                max_attempts=4,
                resource_scales={
                    "memory":  iter([24     , 36     , 300     ]),
                    "runtime": iter([8  * 60, 12 * 60, 60 * 60]),
                },
                location_id=location_id,
                resolution=resolution,
                version=version,
                radius=radius,
            )
        )

    workflow.add_tasks(tasks)
    workflow.bind()

    logger.info(f"Running workflow with ID {workflow.workflow_id}.")
    logger.info("For full information see the Jobmon GUI:")
    logger.info(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")

    status = workflow.run(fail_fast=False)
    logger.info(f"Workflow {workflow.workflow_id} completed with status {status}.")


def worker(
    resolution: str,
    version: str,
    location_id: int,
    radius: int,
    buffer_size: int = 5_000,
):
    pm_data = PopulationModelData()
    model_spec = pm_data.load_model_specification(resolution, version)
    modeling_frame = pm_data.load_modeling_frame(resolution)
    shapes = gpd.read_parquet(f"/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/shapes_lsae_{LSVID}_a2.parquet")
    geometry = shapes.loc[shapes['location_id'] == location_id].geometry.to_crs(modeling_frame.crs)
    buffered_geometry = (
        geometry
        .explode(index_parts=True)
        .convex_hull.buffer(buffer_size)
        .union_all()
    )
    block_keys = modeling_frame.loc[modeling_frame.intersects(buffered_geometry), 'block_key'].unique().tolist()

    time_points = sorted(pm_data.list_raked_prediction_time_points(resolution, version))
    urbanicity = []
    for time_point in tqdm.tqdm(time_points, total=len(time_points)):
        for block_key in block_keys:
            try:
                population = rt.load_raster(
                    pm_data.raked_prediction_path(block_key, time_point, model_spec)
                ).clip(geometry).mask(geometry)
                urban_mask = rt.load_raster(
                    pm_data.model_version_root(resolution, version) / 'urban' / time_point / block_key / f"{radius}m.tif"
                ).clip(geometry).mask(geometry)
                urbanicity.append(
                    pd.DataFrame(
                        {
                            "urban": np.nansum(population * urban_mask),
                            "rural": np.nansum(population) - np.nansum(population * urban_mask),
                        },
                        index=pd.MultiIndex.from_tuples([(location_id, block_key, time_point)], names=['location_id', 'block_key', 'time_point'])
                    )
                )
            except WindowError:
                continue

    urbanicity = pd.concat(urbanicity)
    urbanicity = urbanicity.groupby(['location_id', 'time_point']).sum()
    urbanicity.to_parquet(
        pmc.MODEL_ROOT / "scratch" / SCRATCH_DIR / f"{location_id}.parquet"
    )


def runner(resolution: str, version: str, overwrite: bool, radius: int = 1_000):
    hierarchy = pd.read_parquet(f"/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_lsae_{LSVID}.parquet")
    location_ids = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()

    if not overwrite:
        logger.info("Identifying unwritten location_id files.")
        location_ids = [
            location_id for location_id in tqdm.tqdm(location_ids, total=len(location_ids))
            if not (pmc.MODEL_ROOT / "scratch" / SCRATCH_DIR / f"{location_id}.parquet").exists()
        ]
    
    logger.info(f"Calculating urban/rural populations for {len(location_ids)} admin2 locations.")
    mkdir(pmc.MODEL_ROOT / "scratch" / SCRATCH_DIR, exist_ok=True)
    execute_workflow(
        resolution=resolution,
        version=version,
        location_ids=location_ids,
        radius=radius,
    )


if __name__ == "__main__":
    if sys.argv[1] == "runner":
        if sys.argv[4] not in ["True", "False"]:
            raise ValueError(f"Invalid overwrite flag: {sys.argv[4]}")
        runner(
            resolution=sys.argv[2],
            version=sys.argv[3],
            overwrite=sys.argv[4] == "True",
        )
    elif sys.argv[1] == "worker":
        worker(
            resolution=int(sys.argv[2]),
            version=sys.argv[3],
            location_id=int(sys.argv[4]),
            radius=int(sys.argv[5]),
        )
