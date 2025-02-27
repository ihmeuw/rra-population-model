import shutil

from rra_population_model.data import PopulationModelData


def get_prediction_time_point(
    pm_data: PopulationModelData,
    resolution: str,
    version: str,
    time_point: str,
) -> str:
    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    if time_point in prediction_time_points:
        return time_point
    else:
        target_year = int(time_point.split("q")[0])
        min_year = int(min(prediction_time_points).split("q")[0])
        max_year = int(max(prediction_time_points).split("q")[0])
        load_year = min(max(int(target_year), min_year), max_year)
        return f"{load_year}q1"


def check_gdal_installed() -> None:
    if shutil.which("gdalbuildvrt") is None:
        msg = "gdalbuildvrt not found. Please install GDAL with `conda install conda-forge::gdal`."
        raise ValueError(msg)
