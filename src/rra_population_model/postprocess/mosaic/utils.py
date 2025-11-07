import shlex
import shutil
import functools
import subprocess
from pathlib import Path

from rra_tools.shell_tools import touch
from rra_tools import parallel

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification


def save_change_group_key(
    group_key: str,
    pm_data: PopulationModelData,
    time_points: list[str],
    model_spec: ModelSpecification,
    measure: str,
):
    start_raster = pm_data.load_compiled_prediction(
        group_key,
        time_points[0],
        model_spec,
        measure,
    )
    end_raster = pm_data.load_compiled_prediction(
        group_key,
        time_points[-1],
        model_spec,
        measure,
    )
    change_raster = end_raster - start_raster

    pm_data.save_compiled_prediction(
        raster=change_raster,
        group_key=group_key,
        time_point="-".join(time_points),
        model_spec=model_spec,
        measure="change",
        resampling="average",
    )


def save_change(
    resolution: str,
    version: str,
    time_points: list[str],
    output_dir: str,
    num_cores: int,
    measure: str = "population",
):
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    group_keys = pm_data.list_compiled_prediction_time_point_group_keys(resolution, version, time_points[-1], measure)

    _save_change_group_key = functools.partial(
        save_change_group_key,
        pm_data=pm_data,
        time_points=time_points,
        model_spec=model_spec,
        measure=measure,
    )
    parallel.run_parallel(
        _save_change_group_key,
        group_keys,
        num_cores=num_cores,
        progress_bar=True,
    )


def make_vrt(vrt_path: Path) -> None:
    gdalbuildvrt_path = shutil.which("gdalbuildvrt")
    files = " ".join([str(p) for p in vrt_path.parent.glob("G*.tif")])

    touch(vrt_path, clobber=True)

    cmd = f"{gdalbuildvrt_path} {vrt_path} {files}"

    print("building vrt")
    subprocess.run(shlex.split(cmd), check=True)


def make_vrts(
    time_points: list[str],
    model_spec: ModelSpecification,
    pm_data: PopulationModelData,
    measure: str = "",
) -> None:
    for tp in time_points:
        print(tp)
        vrt_path = pm_data.compiled_prediction_vrt_path(tp, model_spec, measure)
        make_vrt(vrt_path)
