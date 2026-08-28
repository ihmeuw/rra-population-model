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
    get_processing_strategy,
)
from rra_population_model.model_prep.features.metadata import get_feature_metadata
from rra_population_model.model_prep.features.msft_obm import process_msft_obm
from rra_population_model.model_prep.features.ntl import process_ntl
from rra_population_model.model_prep.features.overture import process_overture

# GHSL first, as we need the residential mask for msft. OBM last of the built
# versions, because it reads GHSL's height - and because process_msft_obm,
# which runs after this loop, reads the residential fraction OBM writes.
BUILT_VERSIONS = [
    pmc.BUILT_VERSIONS["ghsl_r2023a"],
    pmc.BUILT_VERSIONS["microsoft_v6"],
    pmc.BUILT_VERSIONS["microsoft_v7"],
    pmc.BUILT_VERSIONS["microsoft_v7_1"],
    pmc.BUILT_VERSIONS["microsoft_v8"],
    pmc.OBM_BUILT_VERSION,
]


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

    for built_version in BUILT_VERSIONS:
        print(f"Processing {built_version.name}")
        strategy, fill_time_points = get_processing_strategy(
            built_version, feature_metadata
        )
        measure_paths = strategy.generate_measures(bd_data, pm_data)
        derived_measure_paths = strategy.generate_derived_measures(bd_data, pm_data)
        strategy.link_features(
            {**measure_paths, **derived_measure_paths},
            fill_time_points,
            feature_metadata,
            pm_data,
        )

    print("Processing NTL")
    process_ntl(
        feature_metadata=feature_metadata,
        pm_data=pm_data,
    )

    print("Processing overture")
    process_overture(
        feature_metadata=feature_metadata,
        pm_data=pm_data,
    )

    # OBM is built by the BUILT_VERSIONS loop above, via ObmStrategy - it is a
    # registered built version rather than a special case.

    # Must follow OBM: it reads the residential fraction OBM just wrote (or the
    # symlink an earlier canonical run left at this time point).
    print("Processing Microsoft v8 with the OBM residential split")
    process_msft_obm(
        pm_data=pm_data,
        resolution=feature_metadata.resolution,
        block_key=feature_metadata.block_key,
        time_point=feature_metadata.time_point,
    )


def geospatial_average_features_main(
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
    features_to_average = [
        "density",
        "volume",
        "nonresidential_density",
        "nonresidential_volume",
        "residential_density",
        "residential_volume",
    ]
    for built_version in BUILT_VERSIONS:
        print(f"Processing {built_version.name}")
        strategy, fill_time_points = get_processing_strategy(
            built_version, feature_metadata
        )
        feature_paths = strategy.generate_geospatial_averages(
            [f"{built_version.name}_{feature}" for feature in features_to_average],
            pmc.FEATURE_AVERAGE_RADII,
            pm_data,
        )
        strategy.link_features(
            feature_paths,
            fill_time_points,
            feature_metadata,
            pm_data,
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
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def geospatial_average_features_task(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    geospatial_average_features_main(
        block_key, time_point, resolution, building_density_dir, output_dir
    )


def obm_features_main(
    block_key: str,
    resolution: str,
    building_density_dir: str | Path,
    model_root: str | Path,
) -> None:
    """Build only the OBM features for one block, at the canonical time point.

    A thin wrapper over the same ObmStrategy the general features command uses,
    so the two cannot produce different rasters. It exists because rebuilding
    OBM alone is common and running the full feature set to do it would also
    rewrite every other provider.
    """
    print(f"Processing OBM features for block {block_key}")
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(model_root)

    built_version = pmc.OBM_BUILT_VERSION
    feature_metadata = get_feature_metadata(
        pm_data, bd_data, resolution, block_key, built_version.time_points[0]
    )
    strategy, fill_time_points = get_processing_strategy(
        built_version, feature_metadata
    )
    measure_paths = strategy.generate_measures(bd_data, pm_data)
    if not measure_paths:
        return
    derived_measure_paths = strategy.generate_derived_measures(bd_data, pm_data)
    strategy.link_features(
        {**measure_paths, **derived_measure_paths},
        fill_time_points,
        feature_metadata,
        pm_data,
    )


@click.command()
@clio.with_block_key()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def obm_features_task(
    block_key: str,
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    """Build the OBM features for a given block."""
    obm_features_main(block_key, resolution, building_density_dir, output_dir)


@click.command()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def obm_features(
    resolution: str,
    building_density_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    """Prepare the Open Building Map features."""
    pm_data = PopulationModelData(output_dir)
    # Only blocks with OBM coverage exist, and that set is already the right one;
    # the modeling frame is a superset.
    block_keys = pm_data.list_open_building_map_blocks(resolution)
    print(f"Submitting {len(block_keys)} jobs to process OBM features")

    jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="obm_features",
        node_args={
            "block-key": block_keys,
        },
        task_args={
            "building-density-dir": building_density_dir,
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            # Eight 8192^2 float32 reads promoted to float64, plus a GHSL height
            # read and several derived arrays of the same size.
            #
            # Sized from the first full 2,902-block run, which peaked at 9.54G
            # and 3 minutes. 11G leaves ~15% headroom over the observed maximum
            # and 5m leaves ~65% - thin on memory, so a block denser than
            # anything in that run would be killed rather than slowed. Slurm
            # enforces memory as a hard cap and the retry would hit the same
            # ceiling, so raise this before assuming a failure is transient.
            "memory": "11G",
            "runtime": "5m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=pm_data.log_dir("preprocess_obm_features"),
        max_attempts=3,
    )


def msft_obm_features_main(
    block_key: str,
    time_point: str,
    resolution: str,
    model_root: str | Path,
) -> None:
    """Build only the Microsoft-v8-with-OBM features for one block/time point."""
    pm_data = PopulationModelData(model_root)
    process_msft_obm(
        pm_data=pm_data,
        resolution=resolution,
        block_key=block_key,
        time_point=time_point,
    )


@click.command()
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_output_directory(pmc.MODEL_ROOT)
def msft_obm_features_task(
    block_key: str,
    time_point: str,
    resolution: str,
    output_dir: str,
) -> None:
    """Build the Microsoft-v8-with-OBM features for a given block."""
    msft_obm_features_main(block_key, time_point, resolution, output_dir)


@click.command()
@clio.with_time_point(
    choices=pmc.BUILT_VERSIONS[pmc.MSFT_V8_SOURCE_PROVIDER].time_points,
    allow_all=True,
)
@clio.with_resolution()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def msft_obm_features(
    time_point: list[str],
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Prepare Microsoft v8 with the OBM residential split."""
    pm_data = PopulationModelData(output_dir)
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()
    # Mirror microsoft_v8: real files at its own 24 quarters only. Everything
    # earlier is back-filled by symlink when the first quarter is built, so
    # asking for all 70 time points would do 46 blocks' worth of needless work.
    epochs = pmc.BUILT_VERSIONS[pmc.MSFT_V8_SOURCE_PROVIDER].time_points
    time_point = [t for t in time_point if t in epochs] or list(epochs)
    print(f"Submitting {len(block_keys) * len(time_point)} jobs "
          f"({len(block_keys):,} blocks x {len(time_point)} v8 epochs)")

    jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="msft_obm_features",
        node_args={"block-key": block_keys, "time-point": time_point},
        task_args={"output-dir": output_dir, "resolution": resolution},
        task_resources={
            "queue": queue,
            "cores": 1,
            # Four 8192^2 rasters live at once: v8 volume plus a density and a
            # fraction for each p source.
            "memory": "12G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=pm_data.log_dir("preprocess_msft_obm_features"),
        max_attempts=3,
    )


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
            "memory": "5G",
            "runtime": "10m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=pm_data.log_dir("preprocess_features"),
        max_attempts=3,
    )
