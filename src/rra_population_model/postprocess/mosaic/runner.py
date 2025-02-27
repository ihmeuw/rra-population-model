import itertools

import click
import rasterra as rt
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.mosaic import utils
from rra_population_model.postprocess.utils import check_gdal_installed

STRIDE = 10


def mosaic_main(
    resolution: str,
    version: str,
    bx: int,
    by: int,
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
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
    pm_data.save_compiled_prediction(
        raster=r,
        group_key=group_key,
        time_point=time_point,
        model_spec=model_spec,
        num_cores=num_cores,
        resampling="average",
    )


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_version()
@click.option("--bx", type=int, required=True)
@click.option("--by", type=int, required=True)
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
def mosaic_task(
    resolution: str,
    version: str,
    bx: int,
    by: int,
    time_point: str,
    output_dir: str,
    num_cores: int,
) -> None:
    mosaic_main(resolution, version, bx, by, time_point, output_dir, num_cores)


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(8)
@clio.with_queue()
def mosaic(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    check_gdal_installed()
    pm_data = PopulationModelData(output_dir)

    raked_time_points = pm_data.list_raked_prediction_time_points(resolution, version)
    time_points = clio.convert_choice(time_point, raked_time_points)

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
        task_name="mosaic",
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": "120G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "bx": bxs,
            "by": bys,
            "time-point": time_points,
        },
        task_args={
            "resolution": resolution,
            "version": version,
            "num-cores": num_cores,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_mosaic"),
    )

    print("Building VRTs")
    model_spec = pm_data.load_model_specification(resolution, version)
    utils.make_vrts(
        time_points,
        model_spec=model_spec,
        pm_data=pm_data,
    )
