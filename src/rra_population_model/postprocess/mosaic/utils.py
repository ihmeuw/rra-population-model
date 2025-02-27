import shlex
import shutil
import subprocess
from pathlib import Path

from rra_tools.shell_tools import touch

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification


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
) -> None:
    for tp in time_points:
        print(tp)
        vrt_path = pm_data.compiled_prediction_vrt_path(tp, model_spec)
        make_vrt(vrt_path)
