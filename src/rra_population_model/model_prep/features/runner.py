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

# GHSL first, as we need the residential mask for msft.
#
# OBM is deliberately absent. It is a single epoch built at 2025q2 and linked
# everywhere else, so no per-time-point task can rely on it existing, and it
# reads GHSL rasters another task writes. It is built by `obm_features`, and
# Microsoft v8 with OBM's residential split by `msft_obm_features`, after this
# command - see `features`.
#
# Keyed by resolution. v6, v7 and v7_1 are gone from pmc.BUILT_VERSIONS, so
# naming them here would raise a KeyError rather than silently skip them.
BUILT_VERSIONS = {
    '40': [
        pmc.BUILT_VERSIONS["ghsl_r2023a"],
        pmc.BUILT_VERSIONS["microsoft_v8"],
    ],
    '100': [
        pmc.BUILT_VERSIONS["ghsl_r2023a"],
        pmc.BUILT_VERSIONS["microsoft_v8"],
    ],
}


def features_main(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str | Path,
    output_dir: str | Path,
) -> None:
    print(f"Processing features for block {block_key} at time {time_point}")
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(output_dir)

    print("Loading all feature metadata")
    feature_metadata = get_feature_metadata(
        pm_data, bd_data, resolution, block_key, time_point
    )

    for built_version in BUILT_VERSIONS[resolution]:
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


def geospatial_average_features_main(
    block_key: str,
    time_point: str,
    resolution: str,
    building_density_dir: str | Path,
    output_dir: str | Path,
) -> None:
    print(f"Processing features for block {block_key} at time {time_point}")
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(output_dir)

    print("Loading all feature metadata")
    feature_metadata = get_feature_metadata(
        pm_data, bd_data, resolution, block_key, time_point
    )
    features_to_average = [
        "density",
        "volume",
        # "nonresidential_density",
        # "nonresidential_volume",
        # "residential_density",
        "residential_volume",
    ]
    for built_version in BUILT_VERSIONS[resolution]:
        print(f"Processing {built_version.name}")
        strategy, fill_time_points = get_processing_strategy(
            built_version, feature_metadata
        )
        built_version_features = [f"{built_version.name}_{feature}" for feature in features_to_average]
        feature_paths = strategy.generate_geospatial_averages(
            built_version_features,
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
    """Build the OBM features for one block, at the canonical time point.

    The only route to OBM's rasters: the general `features` command does not
    build them. Requires `features` first - ObmStrategy reads GHSL's feature
    rasters at `pmc.OBM_GHSL_TIME_POINT` - and must precede
    `msft_obm_features`, which reads what this writes.
    """
    print(f"Processing {pmc.OBM_PROVIDER} features for block {block_key}")
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
    """Prepare the Open Building Map features.

    Run after `features` and before `msft_obm_features`.
    """
    pm_data = PopulationModelData(output_dir)
    # Only blocks with OBM coverage exist, and that set is already the right one;
    # the modeling frame is a superset.
    block_keys = pm_data.list_open_building_map_blocks(resolution)
    print(f"Submitting {len(block_keys)} jobs to process {pmc.OBM_PROVIDER} features")

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
            # Before the label-source split, the first full 2,902-block run
            # peaked at 9.54G and 3 minutes and was sized at 11G / 5m. Slurm
            # enforces memory as a hard cap and the retry would hit the same
            # ceiling, so raise this before assuming a failure is transient.
            #
            # The label-source split doubles the reads (15 layers, not 8) but
            # not the resident arrays, which are still eight parent
            # accumulators; the extra is one layer in flight. All 11,608 tasks
            # of the 2026-10-02 split runs completed first-attempt under 13G:
            # sacct MaxRSS p50 11.7G, max 12.37G; elapsed p50 97s, max 229s.
            # MaxRSS is sampled every 30s, so the true peak may sit above
            # 12.37G, but it is below 13G - 15G leaves at least ~15% headroom.
            # 10m is generous.
            "memory": "15G",
            "runtime": "10m",
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
    """Build the Microsoft-v8-with-OBM features for one block/time point.

    Requires `features` (Microsoft v8, GHSL) and `obm_features` first, and
    raises if either has not run for this block and time point.
    """
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
    """Prepare Microsoft v8 with the OBM residential split.

    The last of the three feature steps: run after `features` and then
    `obm_features`.
    """
    pm_data = PopulationModelData(output_dir)
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()
    # Mirror microsoft_v8: real files at its own 24 quarters only. Everything
    # earlier is back-filled by symlink when the first quarter is built, so
    # asking for all 70 time points would do 46 blocks' worth of needless work.
    epochs = pmc.BUILT_VERSIONS[pmc.MSFT_V8_SOURCE_PROVIDER].time_points
    time_point = [t for t in time_point if t in epochs] or list(epochs)

    # Check OBM up front. Every task would raise on its own, but only after
    # being scheduled, and each is retried - tens of thousands of failed
    # attempts for a step that was simply skipped. The per-task checks stay as
    # the backstop.
    obm_root = pm_data.open_building_map_covariates / f"{resolution}m"
    if not obm_root.is_dir():
        msg = f"No OBM covariate at {resolution}m ({obm_root})."
        raise FileNotFoundError(msg)
    unbuilt = [
        bk
        for bk in pm_data.list_open_building_map_blocks(resolution)
        if not pm_data.feature_exists(
            resolution=resolution,
            block_key=bk,
            feature_name=f"{pmc.OBM_PROVIDER}_proportion_residential",
            time_point=pmc.OBM_TIME_POINT,
        )
    ]
    if unbuilt:
        msg = (
            f"{len(unbuilt):,} OBM-covered blocks have no {pmc.OBM_PROVIDER} "
            f"at {pmc.OBM_TIME_POINT} (e.g. {unbuilt[0]}). Run "
            f"`pmrun model_prep obm_features` first."
        )
        raise FileNotFoundError(msg)

    print(f"Submitting {len(block_keys) * len(time_point)} jobs "
          f"({len(block_keys):,} blocks x {len(time_point)} v8 epochs)")

    jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="msft_obm_features",
        node_args={"block-key": block_keys, "time-point": time_point},
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
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
    """Prepare model features.

    Builds GHSL, Microsoft v8, NTL and Overture. OBM and Microsoft v8 with OBM's
    residential split are separate steps, run in this order:

        features -> obm_features -> msft_obm_features

    Rerunning this command does not refresh the later two; rerun them as well
    if GHSL or Microsoft v8 changed.
    """
    pm_data = PopulationModelData(output_dir)
    print("Loading the modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()

    njobs = len(block_keys) * len(time_point)
    print(f"Submitting {njobs} jobs to process features")

    if "2020q2" in time_point:
        # jobs that do overture processing need more resources
        jobmon.run_parallel(
            runner="pmtask model_prep",
            task_name="features",
            node_args={
                "block-key": block_keys,
                "time-point": ["2020q2"],
            },
            task_args={
                "building-density-dir": building_density_dir,
                "output-dir": output_dir,
                "resolution": resolution,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "66G",
                "runtime": "120m",
                "project": "proj_rapidresponse",
                "constraints": "archive",
            },
            log_root=pm_data.log_dir("preprocess_features"),
            max_attempts=3,
            concurrency_limit=2_500,
        )
    time_point_excl = [tp for tp in time_point if tp != "2020q2"]
    if time_point_excl:
        jobmon.run_parallel(
            runner="pmtask model_prep",
            task_name="features",
            node_args={
                "block-key": block_keys,
                "time-point": time_point_excl,
            },
            task_args={
                "building-density-dir": building_density_dir,
                "output-dir": output_dir,
                "resolution": resolution,
            },
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": "6G",
                "runtime": "6m",
                "project": "proj_rapidresponse",
                "constraints": "archive",
            },
            log_root=pm_data.log_dir("preprocess_features"),
            max_attempts=3,
            concurrency_limit=1_000,
        )


@click.command()
@clio.with_time_point(allow_all=True)
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def geospatial_average_features(
    time_point: list[str],
    resolution: str,
    building_density_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    """Prepare model geospatial average features."""
    pm_data = PopulationModelData(output_dir)
    print("Loading the modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_keys = modeling_frame.block_key.unique().tolist()

    njobs = len(block_keys) * len(time_point)
    print(f"Submitting {njobs} jobs to process geospatial average features")

    jobmon.run_parallel(
        runner="pmtask model_prep",
        task_name="geospatial_average_features",
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
            "memory": "20G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=pm_data.log_dir("preprocess_geospatial_average_features"),
        max_attempts=2,
        concurrency_limit=1_000,
    )
