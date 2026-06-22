import click
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import MergeAlg, rasterize
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.validate.metrics.runner import build_bounds_map


def comparison_validation_main(
    source: str,
    iso3: str,
    year: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Loading comparison data")
    pop_raster = pm_data.load_comparison_data(source, iso3, year)
    pop_raster = pop_raster.set_no_data_value(np.nan)
    pop_arr = pop_raster._ndarray  # noqa: SLF001

    print("Loading and subsetting census data")
    path = pm_data.census_path(iso3, year)
    max_admin_level = int(
        pd.read_parquet(path, columns=["admin_level"]).admin_level.max()
    )
    gdf = gpd.read_parquet(
        path,
        filters=[("admin_level", "==", max_admin_level)],
    )
    gdf = gdf.to_crs(pop_raster.crs)
    # pop_raster = pop_raster.to_crs(gdf.crs)

    print("Calculating pixel metrics")
    shape_values = [(shape, i + 1) for i, shape in enumerate(gdf.geometry)]
    bounds_map = build_bounds_map(pop_raster, shape_values)

    location_mask = np.zeros_like(pop_raster, dtype=np.uint32)
    location_mask = rasterize(
        shape_values,
        out=location_mask,
        transform=pop_raster.transform,
        merge_alg=MergeAlg.replace,
    )
    final_bounds_map = {
        i - 1: (rows, cols, location_mask[rows, cols] == i)
        for i, (rows, cols) in bounds_map.items()
    }

    data = []
    for rows, cols, mask in final_bounds_map.values():
        loc_pop = np.nansum(pop_arr[rows, cols][mask])
        data.append(loc_pop)

    results = gdf[["shape_id"]].copy()
    results["iso3"] = iso3
    results["year"] = year
    results["population"] = data

    pm_data.save_comparison_validation(
        results,
        source,
        iso3,
        year,
    )


@click.command()
@click.option("--source", type=str, required=True)
@clio.with_iso3()
@clio.with_year()
@clio.with_output_directory(pmc.MODEL_ROOT)
def comparison_validation_task(source: str, iso3: str, year: str, output_dir: str) -> None:
    comparison_validation_main(source, iso3, year, output_dir)


@click.command()
@click.option("--source", type=str, required=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def comparison_validation(
    source: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    census_tasks = pm_data.list_census_data()
    census_tasks = [
        i[:2] for i in census_tasks
        if f"{i[1]}q{i[2]}" in pmc.MODELING_TIME_POINTS and int(i[1]) < 2023 and "_" not in i[0]
    ]

    jobmon.run_parallel(
        runner="pmtask validate",
        task_name="comparison_validation",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "5m",
            "project": "proj_rapidresponse",
        },
        flat_node_args=(("iso3", "year"), census_tasks),
        task_args={
            "source": source,
            "output-dir": output_dir,
        },
        max_attempts=3,
        resource_scales={
            "memory":  iter([60     , 240     ]),  # G
            "runtime": iter([10 * 60, 60 * 60 ]),  # seconds
        },
        log_root=pm_data.log_dir("validate_comparison"),
    )
