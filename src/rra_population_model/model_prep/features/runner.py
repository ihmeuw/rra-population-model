from pathlib import Path

import click
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.model_prep.features.built import (
    process_built_measure,
    process_residential_density,
)
from rra_population_model.model_prep.features.metadata import get_feature_metadata
from rra_population_model.model_prep.features.ntl import process_ntl


def features_main(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str | Path,
    model_root: str | Path,
) -> None:
    print(f"Processing features for block {block_key} at time {time_point}")
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(model_root)

    print("Loading all feature metadata")
    feature_metadata = get_feature_metadata(
        pm_data, bd_data, resolution, block_key, time_point
    )

    for built_version in pmc.BUILT_VERSIONS.values():
        for measure in built_version.measures:
            process_built_measure(
                built_version=built_version,
                measure=measure,
                feature_metadata=feature_metadata,
                bd_data=bd_data,
                pm_data=pm_data,
            )

    print("Processing residential density")
    process_residential_density(
        provider="ghsl_r2023a",
        feature_metadata=feature_metadata,
        pm_data=pm_data,
    )

    print("Processing NTL")
    process_ntl(
        feature_metadata=feature_metadata,
        pm_data=pm_data,
    )


@click.command()
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def features_task(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    """Build predictors for a given tile and time point."""
    features_main(block_key, time_point, resolution, building_density_dir, output_dir)


@click.command()
@clio.with_time_point(allow_all=True)
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def features(
    time_point: list[str],
    resolution: str,
    building_density_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    """Prepare model features."""
    pm_data = PopulationModelData(output_dir)
    print("Loading the modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()

    njobs = len(block_keys) * len(time_point)
    print(f"Submitting {njobs} jobs to process features")

    jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="features",
        node_args={
            "block-key": block_keys,
            "time-point": time_point,
        },
        task_args={
            "building-density-dir": building_density_dir,
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "15G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=pm_data.log_dir("preprocess_features"),
        max_attempts=3,
    )
