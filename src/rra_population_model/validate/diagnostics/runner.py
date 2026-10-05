"""CLI for the diagnostics stage: one orchestrator and four task commands.

    pmrun  validate diagnostics --resolution 40 --version V [--compare-version V0] [--lanes a,b,c]
    pmtask validate diagnostics_census    --iso3 DEU --time-point 2022q1 ...
    pmtask validate diagnostics_scan      --time-point 2020q1 --num-cores 8 ...
    pmtask validate diagnostics_attribute --block-key B-... --time-point ALL [--trajectory-sample] ...
    pmtask validate diagnostics_collate   [--compare-version V0] ...

Lane A runs one jobmon task per census, Lane B one scan task per time point and then one
attribute task per flagged or sampled block, and the collator runs in-process at the end
(exit status 1 when a GATE fails). Every workflow gates on its outputs afterwards, because
jobmon.run_parallel returns even when tasks failed.
"""

from __future__ import annotations

import click
import numpy as np
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.validate.diagnostics import artifacts, collate, lane_a, lane_b
from rra_population_model.validate.diagnostics.registry import (
    SAMPLE_BLOCKS,
    TOP_N_PIXELS,
)

PROJECT = "proj_rapidresponse"


def run_lane_a(pm_data: PopulationModelData, resolution: str, version: str, output_dir: str, queue: str) -> None:
    model_spec = pm_data.load_model_specification(resolution, version)
    units = pm_data.load_census_raking_tasks(model_spec)[["iso3", "census_time_point"]].drop_duplicates()
    # A1 is written by every census task, so it is the completion marker (the check_complete idiom)
    todo = [
        (iso3, ctp) for iso3, ctp in units.itertuples(index=False, name=None)
        if not artifacts.check_path(pm_data, resolution, version, "A1", iso3, ctp).exists()
    ]
    print(f"Lane A: {len(todo)} of {len(units)} censuses to run")
    if not todo:
        return
    jobmon.run_parallel(
        runner="pmtask validate",
        task_name="diagnostics_census",
        task_resources={"queue": queue, "cores": 1, "memory": "12G", "runtime": "12m", "project": PROJECT},
        flat_node_args=(("iso3", "time-point"), todo),
        task_args={"resolution": resolution, "version": version, "output-dir": output_dir},
        max_attempts=2,
        resource_scales={
            "memory":  iter([30     ]),  # G
            "runtime": iter([30 * 60]),  # seconds
        },
        log_root=pm_data.log_dir("validate_diagnostics_census"),
        concurrency_limit=500,
    )
    missing = [u for u in todo if not artifacts.check_path(pm_data, resolution, version, "A1", *u).exists()]
    if missing:
        msg = f"Lane A did not complete for {len(missing)} censuses (first: {missing[:10]})"
        raise RuntimeError(msg)


def run_lane_b(
    pm_data: PopulationModelData, resolution: str, version: str, output_dir: str, queue: str, time_points: list[str]
) -> None:
    todo = [tp for tp in time_points if not artifacts.check_path(pm_data, resolution, version, "B1_scan", tp).exists()]
    print(f"Lane B scan: {len(todo)} of {len(time_points)} time points to run")
    if todo:
        jobmon.run_parallel(
            runner="pmtask validate",
            task_name="diagnostics_scan",
            task_resources={"queue": queue, "cores": 8, "memory": "50G", "runtime": "300m", "project": PROJECT},
            node_args={"time-point": todo},
            task_args={"resolution": resolution, "version": version, "output-dir": output_dir, "num-cores": 8},
            max_attempts=1,
            log_root=pm_data.log_dir("validate_diagnostics_scan"),
        )
    missing = [tp for tp in time_points if not artifacts.check_path(pm_data, resolution, version, "B1_scan", tp).exists()]
    if missing:
        msg = f"Lane B scan did not complete for time points {missing}; attribution needs a complete scan"
        raise RuntimeError(msg)

    scan = artifacts.load_check(pm_data, resolution, version, "B1_scan")
    pixels = artifacts.load_check(pm_data, resolution, version, "B1_pixels")
    flagged = set(scan.loc[scan["flag_class"] == "GATE", "block_key"])
    flagged |= set(scan.loc[scan["n_over_ppp_list"] > 0, "block_key"])
    flagged |= set(pixels.nlargest(TOP_N_PIXELS, "value")["block_key"])
    # deterministic stratified sample over the per-block maximum for B6/B9
    block_max = scan.groupby("block_key")["max"].max().dropna().sort_values()
    positions = np.linspace(0, len(block_max) - 1, min(SAMPLE_BLOCKS, len(block_max))).round().astype(int)
    sample = set(block_max.index[positions])
    blocks = sorted(flagged | sample)
    todo_blocks = [
        (bk, bk in sample) for bk in blocks
        if not artifacts.check_path(pm_data, resolution, version, "B1_attr", bk).exists()
    ]
    print(f"Lane B attribute: {len(todo_blocks)} of {len(blocks)} blocks ({len(flagged)} flagged, {len(sample)} sampled)")
    if not todo_blocks:
        return
    resources: dict[str, str | int] = {"queue": queue, "cores": 1, "memory": "36G", "runtime": "30m", "project": PROJECT}
    jobmon.run_parallel(
        runner="pmtask validate",
        task_name="diagnostics_attribute",
        task_resources=resources,
        flat_node_args=(("block-key", "trajectory-sample"), todo_blocks),
        per_task_resources=lambda args: {**resources, "runtime": "90m"} if args[1] else resources,
        task_args={"resolution": resolution, "version": version, "output-dir": output_dir, "time-point": "ALL"},
        max_attempts=2,
        log_root=pm_data.log_dir("validate_diagnostics_attribute"),
        concurrency_limit=1_000,
    )


@click.command()
@clio.with_resolution(allow_all=False)
@clio.with_version()
@click.option("--compare-version", type=click.STRING, default=None, help="Previous version whose diagnostics C1 diffs against.")
@click.option("--lanes", type=click.STRING, default="a,b,c", show_default=True, help="Comma-separated subset of lanes to run.")
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def diagnostics(
    resolution: str,
    version: str,
    compare_version: str | None,
    lanes: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    lane_set = {lane.strip().lower() for lane in lanes.split(",") if lane.strip()}
    raked = pm_data.list_raked_prediction_time_points(resolution, version)
    time_points = sorted(clio.convert_choice(time_point, raked))
    final_rf = set(pm_data.list_raking_factor_time_points(resolution, version, stage="final"))
    missing = [tp for tp in time_points if tp not in final_rf]
    if missing:
        msg = f"No final raking factors for time points {missing}; the diagnostics stage runs after `rake --stage final`."
        raise ValueError(msg)

    if "a" in lane_set:
        run_lane_a(pm_data, resolution, version, output_dir, queue)
    if "b" in lane_set:
        run_lane_b(pm_data, resolution, version, output_dir, queue, time_points)
    gate_failed = collate.collate_main(resolution, version, compare_version, output_dir, lane_set)
    if gate_failed:
        raise SystemExit(1)


@click.command()
@clio.with_iso3()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
def diagnostics_census_task(iso3: str, time_point: str, resolution: str, version: str, output_dir: str) -> None:
    lane_a.census_main(iso3, time_point, resolution, version, output_dir)


@click.command()
@clio.with_time_point()
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_progress_bar()
def diagnostics_scan_task(
    time_point: str, resolution: str, version: str, output_dir: str, num_cores: int, progress_bar: bool
) -> None:
    lane_b.scan_main(time_point, resolution, version, output_dir, num_cores, progress_bar)


@click.command()
@clio.with_block_key()
@clio.with_time_point(choices=None, allow_all=True)
@click.option("--trajectory-sample", type=bool, default=False, show_default=True, help="Also run the B6/B9 passes on this block.")
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
def diagnostics_attribute_task(
    block_key: str, time_point: str, trajectory_sample: bool, resolution: str, version: str, output_dir: str
) -> None:
    lane_b.attribute_main(block_key, time_point, trajectory_sample, resolution, version, output_dir)


@click.command()
@clio.with_resolution()
@clio.with_version()
@click.option("--compare-version", type=click.STRING, default=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def diagnostics_collate_task(resolution: str, version: str, compare_version: str | None, output_dir: str) -> None:
    if collate.collate_main(resolution, version, compare_version, output_dir):
        raise SystemExit(1)
