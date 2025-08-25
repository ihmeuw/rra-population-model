import sys
from pathlib import Path
from loguru import logger
from typing import List

import pandas as pd
import geopandas as gpd
import rasterra as rt

from jobmon.client.tool import Tool
import uuid
import shutil

from rra_tools.shell_tools import mkdir

from rra_population_model.data import PopulationModelData, save_raster


def workflow(
    location_ids: List[int],
    ihme_loc_ids: List[str],
    time_points: List[str],
    resolution: str,
    version: str,
    output_root: Path,
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
            'cores': 2,
            'memory': '2G',
            'runtime': '1m',
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
                         ' {location_id}'
                         ' {time_point}'
                         ' {resolution}'
                         ' {version}',
        node_args=['location_id', 'time_point'],
        task_args=['resolution', 'version'],
        op_args=[],
    )

    ## compile tasks
    tasks = []
    for location_id, ihme_loc_id in zip(location_ids, ihme_loc_ids):
        for time_point in time_points:
            output_path = output_root / ihme_loc_id / f'{time_point}.tif'
            if not output_path.exists():
                tasks.append(
                    task_template.create_task(
                        max_attempts=7,
                        resource_scales={
                            'memory':  iter([20    , 60     , 100    , 200    , 700    , 900    ]),
                            'runtime': iter([5 * 60, 10 * 60, 15 * 60, 30 * 60, 60 * 60, 90 * 60]),
                        },
                        location_id=location_id,
                        time_point=time_point,
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
        "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2021.parquet"
    )
    location_ids = hierarchy.loc[hierarchy['level'] == 3, 'location_id'].to_list()
    ihme_loc_ids = hierarchy.loc[hierarchy['level'] == 3, 'ihme_loc_id'].to_list()

    pm_data = PopulationModelData()

    output_root = pm_data.root / 'country_data' / f'{resolution}m' / version
    mkdir(output_root, exist_ok=True)
    for ihme_loc_id in ihme_loc_ids:
        mkdir(output_root / ihme_loc_id, exist_ok=True)

    time_points = pm_data.list_compiled_prediction_time_points(
        resolution,
        version,
    )
    time_points = list(sorted(time_points))[1:]

    workflow(
        location_ids=location_ids,
        ihme_loc_ids=ihme_loc_ids,
        time_points=time_points,
        resolution=resolution,
        version=version,
        output_root=output_root,
    )


def worker(
    location_id: int,
    time_point: str,
    resolution: str,
    version: str,
    buffer_size: int = 5000,
):
    logger.info(f'{location_id} - {time_point}')

    logger.info('PREPARING METADATA')
    pm_data = PopulationModelData()
    model_spec = pm_data.load_model_specification(resolution, version)
    output_root = pm_data.root / 'country_data' / f'{resolution}m' / version

    hierarchy = pd.read_parquet(
        "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_gbd_2021.parquet"
    )
    ihme_loc_id = hierarchy.set_index('location_id').loc[location_id, 'ihme_loc_id']

    output_path = output_root / ihme_loc_id / f'{time_point}.tif'
    if not output_path.exists():
        logger.info('LOADING GEOMETRY AND CREATING BUFFERED GEOMETRY')
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

        logger.info('LOADING COMPILED COGs')
        raster = rt.load_raster(
            pm_data.compiled_prediction_vrt_path(time_point, model_spec),
            buffered_geometry.bounds,
        ).clip(geometry).mask(geometry)

        logger.info('SAVING COUNTRY RASTER')
        save_raster(raster, output_path)
    else:
        logger.info('COUNTRY RASTER ALREADY EXISTS')


if __name__ == '__main__':
    if sys.argv[1] == 'runner':
        runner(
            resolution=sys.argv[2],
            version=sys.argv[3],
        )
    elif sys.argv[1] == 'worker':
        worker(
            location_id=int(sys.argv[2]),
            time_point=sys.argv[3],
            resolution=sys.argv[4],
            version=sys.argv[5],
        )
