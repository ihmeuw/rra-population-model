import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import shapely

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification


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


def repair_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make invalid geometries valid in place (task_admins stores raw parent
    geometries -- census_rake only repairs the child census geometries -- and an
    invalid geometry makes ``.intersects`` raise a GEOSException)."""
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
    return gdf


def block_census_tasks(
    task_admins: gpd.GeoDataFrame,
    block_geometry: shapely.Polygon | shapely.MultiPolygon,
) -> list[tuple[str, str, str]]:
    """(iso3, task_parent_id, census_time_point) of census admins touching a block.

    Uses the spatial index (this gets called once per block over ~17k admins with
    complex country geometries). Geometries must already be valid (see
    ``repair_invalid_geometries``) or the intersects predicate can raise a
    GEOSException. Returned as plain tuples so callers can ship them to worker
    processes without pickling the admin geometries.
    """
    idx = task_admins.sindex.query(block_geometry, predicate="intersects")
    return list(
        task_admins.iloc[idx]
        .reset_index()
        .loc[:, ["iso3", "task_parent_id", "census_time_point"]]
        .itertuples(index=False, name=None)
    )


def load_block_census_layer(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    block_geometry: shapely.Polygon | shapely.MultiPolygon,
    prediction_time_point: str,
    census_tasks: list[tuple[str, str, str]],
    census_weights: pd.DataFrame,
) -> rt.RasterArray | None:
    """Build the census population layer for one block at one time point.

    This is the single source of truth for the census splice: the weighted sum of
    every contributing raked-census raster over the block. Both the final raking
    factors (which need per-admin sums of the spliced field) and the final rake
    (which splices this layer onto the GBD-raked surface with a ``merge first``)
    must see the exact same layer, so they both call this.

    ``census_tasks`` comes from ``block_census_tasks``. Returns None when no
    census contributes to this block at this time point.
    """
    # A census only brackets a subset of model time points, so it only has a
    # written raster + a weight for those. Restrict to the (iso3,
    # census_time_point) pairs that contribute to THIS prediction time point:
    # with one census per country that's all of them; with several it avoids
    # loading a raster/weight that was never produced for a non-contributing
    # census (which would otherwise be a missing-file / KeyError crash).
    weights_here = census_weights.reset_index()
    contributing = set(
        weights_here.loc[
            weights_here["model_time_point"] == prediction_time_point,
            ["iso3", "census_time_point"],
        ].itertuples(index=False, name=None)
    )
    census_population = []
    for iso3, shape_id, census_time_point in census_tasks:
        if (iso3, census_time_point) not in contributing:
            continue
        raked_census = pm_data.load_raked_census(
            iso3,
            shape_id,
            prediction_time_point,
            census_time_point,
            model_spec,
            bounds=block_geometry.bounds,
        )
        if np.isnan(raked_census.to_numpy()).all():
            # Admins with no predicted pixels write a minimal all-nodata
            # raster; drop it here (it contributes nothing and would only
            # inflate the merge extent). A *missing* tif raises above,
            # surfacing a failed job instead of silently dropping it.
            continue
        raked_census = raked_census.clip(block_geometry).mask(block_geometry)
        census_population.append(
            raked_census
            *
            census_weights.loc[iso3, prediction_time_point, census_time_point].item()
        )
    if not census_population:
        return None
    return rt.merge(census_population, method="sum")


def paste_on_canvas(
    layer: rt.RasterArray, canvas_like: rt.RasterArray
) -> rt.RasterArray:
    """Return ``layer`` re-gridded onto ``canvas_like``'s exact grid.

    The census layer is built from windowed reads over the block bounds and in
    practice lands on the block grid already, but sum/where arithmetic against
    the block raster requires exact alignment -- one cheap merge guarantees it.
    """
    canvas = rt.RasterArray(
        np.full(canvas_like.shape, np.nan, dtype=np.float32),
        transform=canvas_like.transform,
        crs=canvas_like.crs,
        no_data_value=np.nan,
    )
    return rt.merge([layer, canvas], method="first")
