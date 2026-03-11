from pathlib import Path

import click
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.census_rake import utils


def census_rake_main(
    resolution: set,
    version: str,
    iso3: str,
    task_parent_id: str,
    census_time_point: str,
    output_dir: str | Path,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    task_admins, census_weights = pm_data.load_census_raking_inputs(model_spec)
    task_map = task_admins.loc[[iso3], [task_parent_id], :].reset_index().T[0].rename('task_arg')
    census_weights = census_weights.loc[iso3, :, census_time_point]["weight"]
    model_time_points = census_weights.index.to_list()

    census_data, raw_gdf, raw_prediction, scalar_rasters = utils.process_census_data(
        resolution,
        version,
        pm_data,
        task_map,
        census_time_point,
        model_time_points,
    )
    raked_prediction = utils.rake(
        census_data,
        raw_gdf,
        raw_prediction,
    )

    for model_time_point, scalar_raster in zip([model_time_points, scalar_rasters]):
        raked_raster = raked_prediction * scalar_raster * census_weights.loc[model_time_point]
        raked_raster = utils.trim_null_edges(raked_raster)


@click.command()
@clio.with_resolution()
@clio.with_version()
@click.option("--iso3", type=str, required=True)
@click.option("--task-parent-id", type=str, required=True)
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def census_rake_task(
    resolution: str,
    version: str,
    iso3: str,
    task_parent_id: str,
    time_point: str,
    output_dir: str | Path,
) -> None:
    census_rake_main(
        resolution=resolution,
        version=version,
        iso3=iso3,
        task_parent_id=task_parent_id,
        census_time_point=time_point,
        output_dir=output_dir,
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def census_rake(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = clio.convert_choice(time_point, prediction_time_points)

    task_admins, census_weights = utils.generate_census_inputs(pm_data)
    census_weights = census_weights.loc[:, time_points, :]

    pm_data.save_census_raking_inputs(
        task_admins,
        census_weights,
        model_spec
    )

    print(f"Building raking factors for {len(time_points)} time points.")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="census_rake",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "80G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        flat_node_args=(("iso3", "task-parent-id", "census-time-point"), census_tasks),
        task_args={
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_census_rake"),
    )
