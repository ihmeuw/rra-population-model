import sys
from pathlib import Path
from loguru import logger
from typing import List
import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterra as rt
from affine import Affine

from jobmon.client.tool import Tool
import uuid
import shutil

from rra_tools.shell_tools import mkdir

from rra_population_model.data import (
    PopulationModelData,
    save_raster,
)
from rra_population_model.constants import CRSES

COG_LOCATION_IDS = [
    ## countries that are flagged by the near-antimeridian logic but create problems
    # 413,  # Tokelau
]


def workflow(
    location_id_time_points: List[str],
    resolution: str,
    version: str,
):
    wf_uuid = uuid.uuid4()

    tool = Tool(name='pop_ea_countries')

    workflow = tool.create_workflow(
        name=f'pop_ea_countries_{wf_uuid}',
    )

    ## define templates
    task_template = tool.get_task_template(
        default_compute_resources={
            'queue': 'all.q',
            'memory': '5G',
            'runtime': '3m',
            # 'stdout': str(version_root / '_diagnostics' / 'logs' / 'output'),
            # 'stderr': str(version_root / '_diagnostics' / 'logs' / 'error'),
            'project': 'proj_rapidresponse',
            # 'constraints': 'archive',
        },
        template_name='country_pop',
        default_cluster_name='slurm',
        command_template=f'{shutil.which("python")}'
                         f' {Path(__file__)}'
                         ' worker'
                         ' {location_id_time_point}'
                         ' {resolution}'
                         ' {version}',
        node_args=['location_id_time_point'],
        task_args=['resolution', 'version'],
        op_args=[],
    )

    ## compile tasks
    tasks = []
    for location_id_time_point in location_id_time_points:
        tasks.append(
            task_template.create_task(
                max_attempts=6,
                resource_scales={
                    'memory':  iter([50    , 100    , 200    , 500     , 800     ]),
                    'runtime': iter([4 * 60, 10 * 60, 20 * 60, 240 * 60, 360 * 60]),
                },
                location_id_time_point=location_id_time_point,
                resolution=resolution,
                version=version,
            )
        )

    workflow.add_tasks(tasks)
    workflow.bind()

    logger.info(f'Running workflow with ID {workflow.workflow_id}.')
    logger.info('For full information see the Jobmon GUI:')
    logger.info(f'https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}')

    status = workflow.run(fail_fast=False)
    logger.info(f'Workflow {workflow.workflow_id} completed with status {status}.')


def runner(resolution: str, version: str):
    hierarchy = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2023.parquet"
    )
    is_level_3 = hierarchy['level'] == 3
    location_ids = hierarchy.loc[is_level_3, 'location_id'].to_list()
    ihme_loc_ids = hierarchy.loc[is_level_3, 'ihme_loc_id'].to_list()

    pm_data = PopulationModelData()

    output_root = pm_data.root / 'country_data' / f'{resolution}m' / version
    mkdir(output_root, exist_ok=True)
    for ihme_loc_id in ihme_loc_ids:
        mkdir(output_root / ihme_loc_id, exist_ok=True)

    time_points = pm_data.list_compiled_prediction_time_points(
        resolution,
        version,
        measure="population",
    )
    time_points = [time_point for time_point in list(sorted(time_points)) if time_point != "2020q1"]

    possible = 0
    running = 0
    location_id_time_points = []
    for location_id, ihme_loc_id in zip(location_ids, ihme_loc_ids):
        for time_point in time_points:
            possible += 1
            output_path = output_root / ihme_loc_id / f'{time_point}.tif'
            if not output_path.exists():
                location_id_time_points.append(f'{location_id}-{time_point}')
                running += 1
    complete = possible - running

    logger.info(f'Running {running} location-time points ({complete} already complete).')
    workflow(
        location_id_time_points=location_id_time_points,
        resolution=resolution,
        version=version,
    )


def shift_to_antimeridian(raster: rt.RasterArray) -> rt.RasterArray:
    """Shift a global raster by half the world width (no resampling)."""
    if raster.crs != CRSES["equal_area"].code:
        raise ValueError("Transformation being applied to world cylindrical (equal area) only.")
    shift_px = np.abs(CRSES["equal_area"].bounds[0])
    shift_decimals = len(str(shift_px).split('.')[-1])
    if shift_decimals != 2:
        raise ValueError(f"Expected 2 decimals in CRS width, got {shift_decimals}")
    target_crs = CRSES["equal_area_anti_meridian"].to_pyproj()

    if raster.bounds[0] < 0 and raster.bounds[1] > 0:
        raise ValueError("Crosses prime meridian")

    # Update affine transform x origin by half the world width
    t = raster.transform
    if t.c < 0:
        t_c = np.round(shift_px + t.c, shift_decimals)
    else:
        t_c = np.round(t.c - shift_px, shift_decimals)
    new_transform = Affine(t.a, t.b, float(t_c), t.d, t.e, t.f)

    return rt.RasterArray(
        raster.to_numpy(),
        transform=new_transform,
        crs=target_crs,
        no_data_value=raster.no_data_value,
    )


def worker(
    location_id_time_point: str,
    resolution: str,
    version: str,
    buffer_size: int = 5000,
):
    location_id, time_point = location_id_time_point.split('-')
    location_id = int(location_id)
    logger.info(f'{location_id} - {time_point}')

    logger.info('PREPARING METADATA')
    pm_data = PopulationModelData()
    model_spec = pm_data.load_model_specification(resolution, version)
    modeling_frame = pm_data.load_modeling_frame(resolution)
    output_root = pm_data.root / 'country_data' / f'{resolution}m' / version

    hierarchy = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2023.parquet"
    )
    ihme_loc_id = hierarchy.set_index('location_id').loc[location_id, 'ihme_loc_id']

    logger.info('LOADING AND CREATING BUFFERED GEOMETRY')
    shapes = gpd.read_parquet(
        "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/shapes_lsae_1285_a0.parquet"
    )
    geometry = shapes.to_crs("ESRI:54034").set_index('location_id').loc[location_id, 'geometry']
    buffered_geometry = (
        gpd.GeoSeries(geometry)
        .explode(index_parts=True)
        .convex_hull.buffer(buffer_size)
        .union_all()
    )

    if location_id in COG_LOCATION_IDS:
        near_antimeridian = False
    else:
        block_key_x_max = modeling_frame["block_key"].apply(lambda x: int(x.split("X")[0][-4:])).max()
        modeling_frame = modeling_frame.loc[modeling_frame.intersects(buffered_geometry)]
        block_key_x = modeling_frame["block_key"].apply(lambda x: int(x.split("X")[0][-4:]))
        boundary_blocks = 8
        near_antimeridian = (
            (block_key_x <= boundary_blocks)
            | (block_key_x >= block_key_x_max - boundary_blocks)
        ).any()

    if near_antimeridian:
        logger.info('LOADING RAKED PREDICTION BLOCKS AND REPROJECTING DUE TO ANTIMERIDIAN PROXIMITY')
        block_keys = modeling_frame["block_key"].unique().tolist()

        raster = []
        for block_key in tqdm.tqdm(block_keys, total=len(block_keys)):
            block_raster = pm_data.load_raked_prediction(
                block_key, time_point, model_spec
            )
            block_raster = block_raster.clip(geometry).mask(geometry)
            block_raster = shift_to_antimeridian(block_raster)
            raster.append(block_raster)
        raster = rt.merge(raster)
    else:
        logger.info('LOADING COMPILED COGs')
        raster = rt.load_raster(
            pm_data.compiled_prediction_vrt_path(time_point, model_spec, measure="population"),
            buffered_geometry.bounds,
        ).clip(geometry).mask(geometry)

    logger.info('SAVING COUNTRY RASTER')
    output_path = output_root / ihme_loc_id / f'{time_point}.tif'
    save_raster(raster, output_path)


if __name__ == '__main__':
    if sys.argv[1] == 'runner':
        runner(
            resolution=sys.argv[2],
            version=sys.argv[3],
        )
    elif sys.argv[1] == 'worker':
        worker(
            location_id_time_point=sys.argv[2],
            resolution=sys.argv[3],
            version=sys.argv[4],
        )
