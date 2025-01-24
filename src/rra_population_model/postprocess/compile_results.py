import itertools
from pdb import run
import shlex
import shutil
import subprocess
from pathlib import Path

import click
import rasterra as rt
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir, touch

from rra_population_model.data import PopulationModelData, save_raster_to_cog
from rra_population_model import constants as pmc
from rra_population_model import cli_options as clio


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

    paths = []
    for x, y in itertools.product(range(5), range(5)):
        bx_, by_ = 5 * bx + x, 5 * by + y
        block_key = f"B-{bx_:>04}X-{by_:>04}Y"
        p = pm_data.raked_prediction_path(block_key, time_point, model_spec)
        if p.exists():
            # Clean this up. We are spatially complete, but this loop runs over the edge
            # of the dataset, we don't actually have all the tiles.
            paths.append(p)

    print("loading rasters")
    r = rt.load_mf_raster(paths)

    print("writing cog")
    group_key = f"G-{bx:>04}X-{by:>04}Y"
    pm_data.save_compiled(r, group_key, time_point, model_spec, resampling="average", num_cores=num_cores)


@click.command()  # type: ignore[arg-type]
@clio.with_model_name()
@clio.with_resolution()
@click.option("--bx", type=int, required=True)
@click.option("--by", type=int, required=True)
@clio.with_time_point(pmc.ANNUAL_TIME_POINTS)
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
    compile_results_main(model_name, resolution, bx, by, time_point, output_dir, num_cores)


def make_vrt(
    time_point: str,
    compiled_root: Path
) -> None:
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
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, model_name)
    compiled_root = pm_data.compiled_path("G-0000X-0000Y", time_point, model_spec).parent

    vrt_path = make_vrt(time_point, compiled_root)

    run_stamp = "2025_01_22"
    results_root = pm_data.results / run_stamp
    figure_results_root = pm_data.figure_results / run_stamp

    for out_root in [results_root, figure_results_root]:
        print("Linking native resolution")
        parent_dir = out_root / f"{pmc.CRSES['world_cylindrical'].short_name}_{resolution}"
        mkdir(parent_dir, parents=True, exist_ok=True)
        link_path = parent_dir / f"{time_point}.tif"
        if link_path.exists():
            continue
        link_path.symlink_to(vrt_path)

    gdalwarp_path = shutil.which("gdalwarp")
    upsample_specs = [
        # (pmc.CRSES["world_cylindrical"], 250, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 500, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 1000, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 2000, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 4000, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 8000, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 16000, "average", figure_results_root),
        # (pmc.CRSES["world_cylindrical"], 1000, "sum", results_root),
        # (pmc.CRSES["wgs84"], 0.1, "sum", results_root),
        (pmc.CRSES["wgs84"], 0.01, "sum", results_root),
    ]

    for crs, target_resolution, resampling, out_root in upsample_specs:
        parent_dir = out_root / f"{crs.short_name}_{str(target_resolution).replace('.', 'p')}"
        mkdir(parent_dir, parents=True, exist_ok=True)
        out_path = parent_dir / f"{time_point}.tif"
        touch(out_path, clobber=True)
        cmd = (
            f"{gdalwarp_path} {vrt_path} {out_path} "
            f'-t_srs "{crs.proj_string}" '
            f"-tr {target_resolution} {target_resolution} "
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
@clio.with_time_point(pmc.ANNUAL_TIME_POINTS)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
def upsample_task(
    model_name: str,
    resolution: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    upsample_main(model_name, resolution, time_point, output_dir, num_cores)


@click.command()  # type: ignore[arg-type]
@clio.with_model_name()
@clio.with_resolution()
@clio.with_time_point(pmc.ANNUAL_TIME_POINTS, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
@clio.with_queue()
def compile_results(
    model_name: str,
    resolution: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)

    bxs = list(range(10))
    bys = list(range(4))

    print("Compiling")

    # jobmon.run_parallel(
    #     runner="pmtask postprocess",
    #     task_name="compile",
    #     task_resources={
    #         "queue": queue,
    #         "cores": num_cores,
    #         "memory": "30G",
    #         "runtime": "10m",
    #         "project": "proj_rapidresponse",
    #     },
    #     node_args={
    #         "bx": bxs,
    #         "by": bys,
    #         "time-point": time_point,
    #     },
    #     task_args={
    #         "model-name": model_name,
    #         "resolution": resolution,
    #         "num-cores": num_cores,
    #         "output-dir": output_dir,
    #     },
    #     max_attempts=1,
    #     log_root=pm_data.root / "compiled",
    # )

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
        },
        task_args={
            "model-name": model_name,
            "resolution": resolution,
            "output-dir": output_dir,
            "num-cores": 1,
        },
        max_attempts=1,
        log_root=pm_data.root / "compiled",
    )
