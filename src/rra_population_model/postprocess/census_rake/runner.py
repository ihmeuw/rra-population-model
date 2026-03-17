from pathlib import Path

import click
from rra_tools import jobmon

import pandas as pd

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.census_rake import utils


# resolution = "40"
# version = "2026_03_11.001"
# iso3 = "USA"
# task_parent_id = "35059950200"
# census_time_point = "2020q1"
# output_dir = pmc.MODEL_ROOT

def census_rake_main(
    resolution: str,
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
    model_time_points = sorted(census_weights.index.to_list())

    census_data, prediction_data, template_raster = utils.process_census_data(
        resolution,
        version,
        pm_data,
        task_map,
        census_time_point,
        model_time_points,
    )
    raked_rasters = utils.rake(
        census_data,
        prediction_data,
        template_raster,
        census_time_point,
        model_time_points,
    )

    for model_time_point, raked_raster in zip(model_time_points, raked_rasters):
        pm_data.save_raked_census(
            raked_raster,
            iso3,
            task_parent_id,
            model_time_point,
            census_time_point,
            model_spec,
        )


def check_time_points(census_weights: pd.DataFrame) -> None:
    test = census_weights.reset_index()
    test = test.groupby(["iso3", "census_time_point"]).apply(lambda x: (x["model_time_point"] == x["census_time_point"]).any())
    if not test.all():
        raise ValueError(f"Missing raw_predictions for at least one time-point found in census data:\n{test}")


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_iso3()
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
    check_time_points(census_weights)

    census_tasks = task_admins.reset_index().loc[:, ["iso3", "task_parent_id"]].merge(
        census_weights.reset_index().loc[:, ["iso3", "census_time_point"]].drop_duplicates(),
        on="iso3"
    )

    pm_data.save_census_raking_metadata(
        task_admins,
        census_weights,
        census_tasks,
        model_spec
    )
    census_tasks = list(census_tasks.itertuples(index=False, name=None))

    print(f"Building raking factors for {len(census_tasks)} census time-admins.")
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
        max_attempts=2,
        log_root=pm_data.log_dir("postprocess_census_rake"),
    )
