import itertools
import shlex
import shutil
import subprocess
from collections.abc import Collection
from pathlib import Path
from typing import ParamSpec, TypeVar

import click
import rasterra as rt
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir, touch

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

STRIDE = 10

UPSAMPLE_SPECS = {
    "world_cylindrical_250f": (pmc.CRSES["world_cylindrical"], 250, "average"),
    "world_cylindrical_500f": (pmc.CRSES["world_cylindrical"], 500, "average"),
    "world_cylindrical_1000f": (pmc.CRSES["world_cylindrical"], 1000, "average"),
    "world_cylindrical_2000f": (pmc.CRSES["world_cylindrical"], 2000, "average"),
    "world_cylindrical_4000f": (pmc.CRSES["world_cylindrical"], 4000, "average"),
    "world_cylindrical_8000f": (pmc.CRSES["world_cylindrical"], 8000, "average"),
    "world_cylindrical_16000f": (pmc.CRSES["world_cylindrical"], 16000, "average"),
    "world_cylindrical_5000": (pmc.CRSES["world_cylindrical"], 5000, "sum"),
    "world_cylindrical_1000": (pmc.CRSES["world_cylindrical"], 1000, "sum"),
    "wgs84_0p1": (pmc.CRSES["wgs84"], 0.1, "sum"),
    "wgs84_0p01": (pmc.CRSES["wgs84"], 0.01, "sum"),
}

_T = TypeVar("_T")
_P = ParamSpec("_P")


def with_spec_name(
    choices: Collection[str] = list(UPSAMPLE_SPECS),
    *,
    allow_all: bool = False,
) -> clio.ClickOption[_P, _T]:
    return clio.with_choice(
        "spec_name",
        allow_all=allow_all,
        choices=choices,
        help="Specification for upsampling.",
        required=True,
    )


def check_gdal_installed() -> None:
    if shutil.which("gdalbuildvrt") is None:
        msg = "gdalbuildvrt not found. Please install GDAL with `conda install conda-forge::gdal`."
        raise ValueError(msg)


def compile_results_main(
    model_name: str,
    resolution: str,
    bx: int,
    by: int,
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, model_name)
    block_keys = pm_data.load_modeling_frame(resolution)["block_key"].unique()

    paths = []
    for x, y in itertools.product(range(STRIDE), range(STRIDE)):
        bx_, by_ = STRIDE * bx + x, STRIDE * by + y
        block_key = f"B-{bx_:>04}X-{by_:>04}Y"
        if block_key not in block_keys:
            continue
        paths.append(pm_data.raked_prediction_path(block_key, time_point, model_spec))

    print("loading rasters")
    r = rt.load_mf_raster(paths)

    print("writing cog")
    group_key = f"G-{bx:>04}X-{by:>04}Y"
    pm_data.save_compiled(
        r, group_key, time_point, model_spec, resampling="average", num_cores=num_cores
    )


@click.command()  # type: ignore[arg-type]
@clio.with_model_name()
@clio.with_resolution()
@click.option("--bx", type=int, required=True)
@click.option("--by", type=int, required=True)
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
def compile_results_task(
    model_name: str,
    resolution: str,
    bx: int,
    by: int,
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    compile_results_main(
        model_name, resolution, bx, by, time_point, output_dir, num_cores
    )


def make_vrt(time_point: str, compiled_root: Path) -> Path:
    gdalbuildvrt_path = shutil.which("gdalbuildvrt")
    files = " ".join([str(p) for p in compiled_root.glob(f"G*{time_point}.tif")])

    vrt_path = compiled_root / f"{time_point}.vrt"
    touch(vrt_path, clobber=True)

    cmd = f"{gdalbuildvrt_path} {vrt_path} {files}"

    print("building vrt")
    subprocess.run(shlex.split(cmd), check=True)
    return vrt_path


def upsample_main(
    model_name: str,
    resolution: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, model_name)
    compiled_root = pm_data.compiled_path(
        "G-0000X-0000Y", time_point, model_spec
    ).parent
    vrt_path = compiled_root / f"{time_point}.vrt"

    run_stamp = "2025_01_26"
    results_root = pm_data.results / run_stamp
    figure_results_root = pm_data.figure_results / run_stamp

    if spec_name == "world_cylindrical_1000":
        for out_root in [results_root, figure_results_root]:
            print("Linking native resolution")
            parent_dir = (
                out_root / f"{pmc.CRSES['world_cylindrical'].short_name}_{resolution}"
            )
            mkdir(parent_dir, parents=True, exist_ok=True)
            link_path = parent_dir / f"{time_point}.tif"
            if link_path.exists():
                continue
            link_path.symlink_to(vrt_path)

    gdalwarp_path = shutil.which("gdalwarp")

    crs, target_resolution, resampling = UPSAMPLE_SPECS[spec_name]
    out_root = figure_results_root if "f" in spec_name else results_root

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
        f"-wm 2048 --debug on "
    )
    if resampling == "sum":
        cmd += "-ovr NONE "
    print("warping to ", target_resolution)
    subprocess.run(shlex.split(cmd), check=True)


@click.command()  # type: ignore[arg-type]
@clio.with_model_name()
@clio.with_resolution()
@with_spec_name()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def upsample_task(
    model_name: str,
    resolution: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
) -> None:
    upsample_main(model_name, resolution, spec_name, time_point, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_model_name()
@clio.with_resolution()
@with_spec_name(allow_all=True)
@clio.with_time_point([str(y) for y in range(1950, 1976)], allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
@clio.with_queue()
def compile_results(
    model_name: str,
    resolution: str,
    spec_name: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)
    model_frame = pm_data.load_modeling_frame(resolution)
    x_max = max(
        [
            int(bk.split("-")[1].split("X")[0])
            for bk in model_frame["block_key"].unique()
        ]
    )
    y_max = max(
        [
            int(bk.split("-")[2].split("Y")[0])
            for bk in model_frame["block_key"].unique()
        ]
    )

    bxs = list(range(x_max // STRIDE + int(bool(x_max % STRIDE))))
    bys = list(range(y_max // STRIDE + int(bool(y_max % STRIDE))))

    print("Compiling")

    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="compile",
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": "120G",
            "runtime": "10m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "bx": bxs,
            "by": bys,
            "time-point": time_point,
        },
        task_args={
            "model-name": model_name,
            "resolution": resolution,
            "num-cores": num_cores,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.root / "compiled",
    )

    print("Building VRTs")
    model_spec = pm_data.load_model_specification(resolution, model_name)
    for tp in time_point:
        print(tp)
        compiled_root = pm_data.compiled_path("G-0000X-0000Y", tp, model_spec).parent

        make_vrt(tp, compiled_root)

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
            "time-point": time_point,
            "spec-name": spec_name,
        },
        task_args={
            "model-name": model_name,
            "resolution": resolution,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.root / "compiled",
    )
