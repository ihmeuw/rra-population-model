import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger

import pandas as pd
import numpy as np
import rasterra as rt
import geopandas as gpd
import shapely
from rasterio.features import rasterize

from jobmon.client.tool import Tool
import uuid
import shutil
from rra_tools.shell_tools import mkdir

from rra_population_model import constants as pmc
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
    save_raster,
)
from rra_population_model.model_prep.features.metadata import FeatureMetadata
from rra_population_model.model_prep.features import utils
from rra_population_model.model_prep.features.metadata import get_feature_metadata

POVERTY_RATES_PATH = Path("/mnt/share/geospatial/lsae/economic_indicators/ldi/output/20251022_090128/raked_admin2_urban_rural_poverty.csv")


def execute_workflow(
    resolution: str,
    version: str,
    block_keys: List[str],
    time_points: List[str],
    radii: List[int],
):
    wf_uuid = uuid.uuid4()

    tool = Tool(name="poverty")

    workflow = tool.create_workflow(
        name=f"poverty_{wf_uuid}",
    )

    ## define templates
    task_template = tool.get_task_template(
        default_compute_resources={
            "queue": "all.q",
            "cores": 1,
            "memory": "33G",
            "runtime": "5m",
            "project": "proj_rapidresponse",
        },
        template_name="poverty",
        default_cluster_name="slurm",
        command_template=f"{shutil.which('python')}"
                         f" {Path(__file__)}"
                         " worker"
                         " {resolution}"
                         " {version}"
                         " {block_key}"
                         " {time_point}"
                         " {radius}",
        node_args=["block_key", "time_point", "radius"],
        task_args=["resolution", "version"],
        op_args=[],
    )

    ## compile tasks
    tasks = []
    for block_key in block_keys:
        for time_point in time_points:
            for radius in radii:
                tasks.append(
                    task_template.create_task(
                        max_attempts=2,
                        # resource_scales={
                        #     "memory":  iter([20    , 60     , 100    , 200    , 700    , 900    ]),
                        #     "runtime": iter([5 * 60, 10 * 60, 15 * 60, 30 * 60, 60 * 60, 90 * 60]),
                        # },
                        block_key=block_key,
                        time_point=time_point,
                        resolution=resolution,
                        version=version,
                        radius=radius,
                    )
                )

    workflow.add_tasks(tasks)
    workflow.bind()

    logger.info(f"Running workflow with ID {workflow.workflow_id}.")
    logger.info("For full information see the Jobmon GUI:")
    logger.info(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")

    status = workflow.run(fail_fast=False)
    logger.info(f"Workflow {workflow.workflow_id} completed with status {status}.")


def load_density_mosaic_tile(
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
) -> rt.RasterArray:
    tiles = []
    for buffer_block_key, bounds in feature_metadata.block_bounds.items():
        tile = pm_data.load_raked_prediction(
            block_key=buffer_block_key,
            model_spec=model_spec,
            time_point=feature_metadata.time_point,
            subset_bounds=bounds,
        )
        area = (
            tile.to_gdf()
            .area
            .to_numpy()
            .astype(np.float32)
            .reshape(tile.shape)
        )
        area = rt.RasterArray(
            data=area,
            transform=tile.transform,
            crs=tile.crs,
            no_data_value=np.nan,
        )
        tile /= area  # turn into population density
        try:
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
        except ValueError:
            # This is kind of a hack, but there"s not a clean way to fix it easily.
            # The issue is that the resolution of the tiles do not exactly line up
            # to the bounds of the world as defined by the CRS. The southernmost
            # have one fewer row of pixels as a whole row would extend past the
            # southern edge of the world, causing reprojection issues. The problem
            # is that we read the tile with bounds, and those bounds cause the underlying
            # rasterio to fill in that missing row. Here we just remove the last row
            # of pixels and reproject the tile.
            tile._ndarray = tile._ndarray[:-1].copy()  # noqa: SLF001
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
        tiles.append(tile)

    buffered_measure = utils.suppress_noise(rt.merge(tiles))

    return buffered_measure


def load_poverty_rates(time_point: str, tile: rt.RasterArray) -> Dict[str, rt.RasterArray]:
    year_id, quarter = [int(i) for i in time_point.split("q")]

    poverty_rates = pd.read_csv(POVERTY_RATES_PATH).set_index(["year_id", "location_id"])
    poverty_rates = poverty_rates.loc[:, ["urban_poverty_rate", "rural_poverty_rate"]]
    poverty_rates = poverty_rates.rename(columns={"urban_poverty_rate": "urban", "rural_poverty_rate": "rural"})
    admin2_shapes = gpd.read_parquet(pmc.MODEL_ROOT / "admin-inputs" / "raking" / "gbd-inputs" / "shapes_lsae_1285_a2.parquet").set_index("location_id")

    year_id = min(
        year_id,
        poverty_rates.index.get_level_values("year_id").max()
    )
    next_year_id = min(
        year_id + 1,
        poverty_rates.index.get_level_values("year_id").max()
    )
    weight = (quarter - 1) / 4
    poverty_rates = (
        poverty_rates.loc[year_id] * (1 - weight)
        + poverty_rates.loc[next_year_id] * weight
    )
    poverty_rates = admin2_shapes.join(poverty_rates, how="left").fillna(0)
    poverty_rates = poverty_rates.to_crs(tile.crs)
    
    tile_bounds = (
        tile.bounds[0],
        tile.bounds[2],
        tile.bounds[1],
        tile.bounds[3],
    )
    poverty_rates = (
        poverty_rates
        .overlay(
            gpd.GeoDataFrame({"geometry":[shapely.box(*tile_bounds)]}, crs=tile.crs),
            how="intersection",
            keep_geom_type=True,
        )
    )
    poverty_rates_rasters = {}
    for urbanicity in ["urban", "rural"]:
        shapes = (
            (geom, val) for geom, val in zip(poverty_rates.geometry, poverty_rates[urbanicity])
        )
        poverty_rates_array = rasterize(
            shapes=shapes,
            out_shape=(tile.height, tile.width),
            transform=tile.transform,
            fill=0,
            dtype=np.float32,
        )
        poverty_rates_rasters[urbanicity] = rt.RasterArray(
            np.where(tile.no_data_mask, tile.no_data_value, poverty_rates_array).astype(np.float32),
            transform=tile.transform,
            crs=tile.crs,
            no_data_value=tile.no_data_value,
        )

    return poverty_rates_rasters


def worker(
    resolution: str,
    version: str,
    block_key: str, 
    time_point: str,
    radius: int,
    people_per_km_threshold: int = 300,
):
    pm_data = PopulationModelData()
    bd_data = BuildingDensityData()
    model_spec = pm_data.load_model_specification(resolution, version)

    logger.info("Loading all feature metadata")
    feature_metadata = get_feature_metadata(
        pm_data, bd_data, resolution, block_key, time_point
    )
    if radius > max(pmc.FEATURE_AVERAGE_RADII):
        raise ValueError(f"Specified radius ({radius}) is larger than max radius in module {max(pmc.FEATURE_AVERAGE_RADII)}.")
    
    logger.info("Loading buffered population density tile")
    buffered_tile = load_density_mosaic_tile(feature_metadata, pm_data, model_spec)

    logger.info("Calculating spatial average")
    average_measure = (
        utils.make_spatial_average(
            tile=buffered_tile,
            radius=radius,
            kernel_type="uniform",
            apply_floor=False,
        )
        .resample_to(feature_metadata.block_template, "average")
        .astype(np.float32)
    )

    tile = pm_data.load_raked_prediction(
        block_key=block_key,
        model_spec=model_spec,
        time_point=feature_metadata.time_point,
    )
    average_measure = rt.RasterArray(
        np.where(tile.no_data_mask, tile.no_data_value, average_measure),
        transform=tile.transform,
        crs=tile.crs,
        no_data_value=tile.no_data_value,
    )

    time_points = sorted(pm_data.list_raked_prediction_time_points(resolution, version))
    if time_point in [time_points[0], time_points[-1]]:
        logger.info("Saving population density")
        population_density_path = Path(model_spec.output_root) / "population_density" / time_point / block_key
        if not population_density_path.exists():
            mkdir(population_density_path, parents=True, exist_ok=True)
        save_raster(
            average_measure,
            population_density_path / f"{radius}m.tif"
        )

    logger.info("Defining and saving urban mask")
    threshold = people_per_km_threshold / 1_000 ** 2
    urban = average_measure.to_numpy() >= threshold
    rural = ~urban
    urban = rt.RasterArray(
        np.where(tile.no_data_mask, tile.no_data_value, urban).astype(np.float32),
        transform=tile.transform,
        crs=tile.crs,
        no_data_value=tile.no_data_value,
    )
    rural = rt.RasterArray(
        np.where(tile.no_data_mask, tile.no_data_value, rural).astype(np.float32),
        transform=tile.transform,
        crs=tile.crs,
        no_data_value=tile.no_data_value,
    )
    urban_path = Path(model_spec.output_root) / "urban" / time_point / block_key
    if not urban_path.exists():
        mkdir(urban_path, parents=True, exist_ok=True)
    save_raster(
        urban,
        urban_path / f"{radius}m.tif"
    )

    logger.info("Calculating and saving impoverished population")
    poverty_rates = load_poverty_rates(time_point, tile)
    poverty = tile * (urban * poverty_rates["urban"] + rural * poverty_rates["rural"])
    poverty_path = Path(model_spec.output_root) / "poverty" / time_point / block_key
    if not poverty_path.exists():
        mkdir(poverty_path, parents=True, exist_ok=True)
    save_raster(
        poverty,
        poverty_path / f"{radius}m.tif"
    )


def runner(resolution: str, version: str, radii: List[int] = [1_000, 5_000]):
    pm_data = PopulationModelData()
    modeling_frame = pm_data.load_modeling_frame(resolution)

    block_keys = modeling_frame["block_key"].unique().tolist()
    time_points = sorted(pm_data.list_raked_prediction_time_points(resolution, version))

    logger.info(f"Estimating urbanicity and poverty for {len(block_keys) * len(time_points) * len(radii)} block-key, time-point, radius combinations.")
    execute_workflow(
        resolution=resolution,
        version=version,
        block_keys=block_keys,
        time_points=time_points,
        radii=radii,
    )


if __name__ == "__main__":
    if sys.argv[1] == "runner":
        runner(
            resolution=sys.argv[2],
            version=sys.argv[3],
        )
    elif sys.argv[1] == "worker":
        worker(
            resolution=sys.argv[2],
            version=sys.argv[3],
            block_key=sys.argv[4],
            time_point=sys.argv[5],
            radius=int(sys.argv[6]),
        )
