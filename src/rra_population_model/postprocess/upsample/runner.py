import shlex
import shutil
import subprocess
from collections.abc import Callable, Collection
from pathlib import Path

import click
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir, touch

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import check_gdal_installed

UPSAMPLE_SPECS = {
    "world_cylindrical_250f": (pmc.CRSES["world_cylindrical"], 250, "average"),
    "world_cylindrical_500f": (pmc.CRSES["world_cylindrical"], 500, "average"),
    "world_cylindrical_1000f": (pmc.CRSES["world_cylindrical"], 1000, "average"),
    "world_cylindrical_2000f": (pmc.CRSES["world_cylindrical"], 2000, "average"),
    "world_cylindrical_4000f": (pmc.CRSES["world_cylindrical"], 4000, "average"),
    "world_cylindrical_8000f": (pmc.CRSES["world_cylindrical"], 8000, "average"),
    "world_cylindrical_10000f": (pmc.CRSES["world_cylindrical"], 10000, "average"),
    "world_cylindrical_16000f": (pmc.CRSES["world_cylindrical"], 16000, "average"),
    "world_cylindrical_1000": (pmc.CRSES["world_cylindrical"], 1000, "sum"),
    "world_cylindrical_5000": (pmc.CRSES["world_cylindrical"], 5000, "sum"),
    "world_cylindrical_10000": (pmc.CRSES["world_cylindrical"], 10000, "sum"),
    "wgs84_0p1": (pmc.CRSES["wgs84"], 0.1, "sum"),
    "wgs84_0p01": (pmc.CRSES["wgs84"], 0.01, "sum"),
}


def with_spec_name[**P, T](
    choices: Collection[str] = list(UPSAMPLE_SPECS),
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return clio.with_choice(
        "spec_name",
        allow_all=allow_all,
        choices=choices,
        help="Specification for upsampling.",
        required=True,
    )


def with_run_stamp[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--run-stamp",
        type=str,
        help="Run stamp for the results.",
        required=True,
    )


def link_native_resolution(
    resolution: str,
    spec_name: str,
    time_point: str,
    vrt_path: Path,
    out_root: Path,
) -> None:
    if not spec_name.startswith("world_cylindrical_1000"):
        return

    print("Linking native resolution")
    parent_dir = out_root / f"{pmc.CRSES['world_cylindrical'].short_name}_{resolution}"
    mkdir(parent_dir, parents=True, exist_ok=True)
    link_path = parent_dir / f"{time_point}.tif"
    if link_path.exists():
        link_path.unlink()
    link_path.symlink_to(vrt_path)


def upsample_main(
    run_stamp: str,
    resolution: str,
    version: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    gdalwarp_path = shutil.which("gdalwarp")
    vrt_path = pm_data.compiled_prediction_vrt_path(time_point, model_spec)

    if "f" in spec_name:
        out_root = pm_data.figure_results / run_stamp
    else:
        out_root = pm_data.results / run_stamp
    mkdir(out_root, exist_ok=True)

    link_native_resolution(resolution, spec_name, time_point, vrt_path, out_root)

    crs, target_resolution, resampling = UPSAMPLE_SPECS[spec_name]

    parent_dir = (
        out_root / f"{crs.short_name}_{str(target_resolution).replace('.', 'p')}"
    )
    mkdir(parent_dir, parents=True, exist_ok=True)
    out_path = parent_dir / f"{time_point}.tif"
    touch(out_path, clobber=True)

    xmin, ymin, xmax, ymax = crs.bounds
    cmd = (
        f"{gdalwarp_path} {vrt_path} {out_path} "
        f'-t_srs "{crs.proj_string}" '
        f"-tr {target_resolution} {target_resolution} "
        f"-te {xmin} {ymin} {xmax} {ymax} "
        f"-r {resampling} "
        f"-wm 2048 "
    )
    if resampling == "sum":
        cmd += "-ovr NONE "
    print("warping to ", target_resolution)
    subprocess.run(shlex.split(cmd), check=True)


@click.command()
@with_run_stamp()
@clio.with_resolution()
@clio.with_version()
@with_spec_name()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def upsample_task(
    run_stamp: str,
    resolution: str,
    version: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
) -> None:
    upsample_main(run_stamp, resolution, version, spec_name, time_point, output_dir)


@click.command()
@with_run_stamp()
@clio.with_resolution()
@clio.with_version()
@with_spec_name(allow_all=True)
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
@clio.with_queue()
def upsample(
    run_stamp: str,
    resolution: str,
    version: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)

    compiled_time_points = pm_data.list_compiled_prediction_time_points(
        resolution, version
    )
    # compiled_time_points = [f"{y}q1" for y in range(1950, 1976)]
    time_points = clio.convert_choice(time_point, compiled_time_points)
    time_points = [time_point for time_point in time_points if time_point.startswith('202')]

    print("Upsampling")

    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="upsample",
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": "200G",
            "runtime": "480m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "time-point": time_points,
            "spec-name": spec_name,
        },
        task_args={
            "run-stamp": run_stamp,
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_upsample"),
    )
