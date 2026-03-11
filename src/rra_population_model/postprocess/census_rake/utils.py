from typing import Any

from affine import Affine
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
from shapely import set_precision

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData


def safe_divide(
    a: npt.NDArray[np.floating[Any]] | pd.DataFrame,
    b: npt.NDArray[np.floating[Any]] | pd.DataFrame,
) -> npt.NDArray[np.floating[Any]]:
    """Divide two arrays, but return 0 where both arrays are 0."""

    if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(  # type: ignore[arg-type]
        b.dtype,  # type: ignore[arg-type]
        np.floating,
    ):
        msg = "Both arrays must be floating point."
        raise TypeError(msg)

    mask = ~((a == 0) & (b == 0))
    r = np.zeros_like(a)
    r[mask] = a[mask] / b[mask]
    return r


def _get_valid_data_mask(array: npt.NDArray[np.floating[Any]], no_data_value: Any) -> npt.NDArray[np.bool_]:
    valid_mask = ~np.isnan(array)
    try:
        if not np.isnan(float(no_data_value)):
            valid_mask &= array != no_data_value
    except (TypeError, ValueError):
        pass

    return valid_mask


def _trim_window_from_mask(valid_mask: npt.NDArray[np.bool_]) -> tuple[slice, slice]:
    valid_rows = np.where(valid_mask.any(axis=1))[0]
    valid_cols = np.where(valid_mask.any(axis=0))[0]

    if valid_rows.size == 0 or valid_cols.size == 0:
        msg = "Raster contains no valid pixels after masking."
        raise ValueError(msg)

    row_slice = slice(valid_rows.min(), valid_rows.max() + 1)
    col_slice = slice(valid_cols.min(), valid_cols.max() + 1)

    return row_slice, col_slice


def trim_null_edges(
    raster: rt.RasterArray,
    row_slice: slice | None = None,
    col_slice: slice | None = None,
) -> rt.RasterArray:
    """Drop fully-null outer rows/columns and update the raster transform."""

    array = raster.to_numpy().astype(np.float32)

    if row_slice is None or col_slice is None:
        valid_mask = _get_valid_data_mask(array, raster.no_data_value)
        row_slice, col_slice = _trim_window_from_mask(valid_mask)

    trimmed = array[row_slice, col_slice]
    new_transform = raster.transform * Affine.translation(col_slice.start, row_slice.start)

    return rt.RasterArray(
        data=trimmed,
        transform=new_transform,
        crs=raster.crs,
        no_data_value=raster.no_data_value,
    )


def calculate_tp_weights(census_years: pd.DataFrame) -> pd.DataFrame:
    census_years = pd.concat(
        [
            pd.concat([
                census_years, pd.Series(time_point, name="model_time_point", index=census_years.index)
            ], axis=1)
            for time_point in pmc.MODELING_TIME_POINTS
        ]
    )

    census_years["distance"] = census_years.apply(
        lambda x: 
            (int(x["census_time_point"].split("q")[0]) + int(x["census_time_point"].split("q")[1]) / 4)
            -
            (int(x["model_time_point"].split("q")[0]) + int(x["model_time_point"].split("q")[1]) / 4)
        ,
        axis=1
    )

    census_weights = pd.concat(
        [
            (
                census_years
                .loc[census_years["distance"] <= 0]
                .sort_values("distance", ascending=False)
                .groupby(["iso3", "model_time_point"])[["census_time_point", "distance"]]
                .first()
                .reset_index()
            ),
            (
                census_years
                .loc[census_years["distance"] >= 0]
                .sort_values("distance")
                .groupby(["iso3", "model_time_point"])[["census_time_point", "distance"]]
                .first()
                .reset_index()
            ),
        ]
    )
    census_weights = census_weights.drop_duplicates().set_index(["iso3", "model_time_point"])
    census_weights["distance"] = census_weights["distance"].abs()
    census_weights["distance"] = census_weights.groupby(["iso3", "model_time_point"])["distance"].transform(sum) - census_weights["distance"]
    census_weights["weight"] = (census_weights["distance"] / census_weights.groupby(["iso3", "model_time_point"])["distance"].transform(sum)).fillna(1)
    census_weights = (
        census_weights
        .drop("distance", axis=1)
        .set_index("census_time_point", append=True)
        .sort_index()
    )

    return census_weights


def generate_census_inputs(pm_data: PopulationModelData) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    available_census_years = pm_data.list_census_data()

    keep_iso3s = ["BRA", "MEX", "USA", "ZAF"]
    data_years = [
        i for i in available_census_years
        if i[0] in keep_iso3s
    ]

    task_admins = []
    census_years = []
    for iso3, year, quarter in data_years:
        print(f"Processing {iso3} {year}q{quarter}")
        admins = pm_data.load_census_data(iso3, year)
        admins["iso3"] = iso3
        if iso3 == "USA":
            task_level = 3
        else:
            task_level = min(2, admins.admin_level.max())
        parent_admins = admins.loc[admins.admin_level == task_level]
        parent_admins = parent_admins.loc[:, ["iso3", "shape_id", "admin_level", "geometry"]]
        parent_admins = parent_admins.rename(
            columns={
                "shape_id": "task_parent_id",
                "admin_level": "task_parent_level",
                "geometry": "task_parent_geometry",
            },
        )
        parent_admins = parent_admins.set_index(["iso3", "task_parent_id", "task_parent_level"])

        admins = admins.loc[admins.admin_level == admins.admin_level.max()]
        admins["task_parent_id"] = admins["path_to_top_parent"].str.split(",").str[task_level]
        admins["task_parent_level"] = task_level
        admins["census_time_point"] = f"{year}q{quarter}"
        admins = admins.loc[:, ["iso3", "task_parent_id", "task_parent_level"]].value_counts().rename("most_detailed_units")
        admins = parent_admins.join(admins)
        task_admins.append(admins)
        census_years.append(
            pd.DataFrame({"census_time_point": f"{year}q{quarter}"}, index=pd.Index([iso3], name="iso3"))
        )
    task_admins = pd.concat(task_admins)
    census_years = pd.concat(census_years)

    census_weights = calculate_tp_weights(census_years)

    return task_admins, census_weights


def process_census_data(
    resolution: str,
    version: str,
    pm_data: PopulationModelData,
    task_map: gpd.GeoSeries,
    census_time_point: str,
    model_time_points: list[str],
    buffer_size: int = 1000,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, rt.RasterArray, list[rt.RasterArray]]:
    modeling_frame = pm_data.load_modeling_frame(resolution)
    model_spec = pm_data.load_model_specification(resolution, version)

    census_data = pm_data.load_census_data(task_map["iso3"], census_time_point.split("q")[0])
    # parent_geometry = census_data.loc[census_data["shape_id"] == task_map["task_parent_id"]].geometry
    buffered_parent_geometry = census_data.loc[census_data["shape_id"] == task_map["task_parent_id"]].geometry.buffer(buffer_size)

    census_data = census_data.loc[
        census_data["admin_level"] == census_data["admin_level"].max()
    ]
    census_data = census_data.loc[
        census_data["path_to_top_parent"].apply(lambda x: x.split(",")[task_map["task_parent_level"]] == task_map["task_parent_id"])
    ]
    census_data = census_data.loc[:, ["shape_id", "population_total", "geometry"]]
    census_data = census_data.to_crs(modeling_frame.crs)

    invalid_admins = ~census_data.geometry.is_valid
    if invalid_admins.any():
        census_data.loc[invalid_admins, "geometry"] = census_data.loc[invalid_admins, "geometry"].make_valid()
    if not census_data.geometry.is_valid.all():
        raise ValueError("Invalid admins remain")
    census_data["geometry"] = census_data.geometry.map(lambda g: set_precision(g, 0.001))

    block_keys = (
        modeling_frame.
        loc[modeling_frame.intersects(buffered_parent_geometry.item()), "block_key"]
        .unique()
        .tolist()
    )

    raw_prediction = rt.merge([
        pm_data.load_raw_prediction(block_key, census_time_point, model_spec)
        for block_key in block_keys
    ])
    raw_prediction = raw_prediction.clip(buffered_parent_geometry).mask(buffered_parent_geometry)
    raw_gdf = (
        raw_prediction.to_gdf()
        .reset_index()
        .rename(columns={"value": "pixel_population", "index": "pixel_id"})
        .sort_values("pixel_id")
    )
    raw_gdf["geometry"] = raw_gdf.geometry.map(lambda g: set_precision(g, 0.001))
    raw_gdf["pixel_area"] = raw_gdf.area

    scalar_rasters = []
    for model_time_point in model_time_points:
        scalar_prediction = rt.merge([
            pm_data.load_raw_prediction(block_key, model_time_point, model_spec)
            for block_key in block_keys
        ])
        scalar_prediction = scalar_prediction.clip(buffered_parent_geometry).mask(buffered_parent_geometry)
        scalar_prediction = safe_divide(scalar_prediction.to_numpy(), raw_prediction.to_numpy())
        scalar_rasters.append(
            rt.RasterArray(
                data=scalar_prediction,
                transform=raw_prediction.transform,
                crs=raw_prediction.crs,
                no_data_value=np.nan,
            )
        )

    return census_data, raw_gdf, raw_prediction, scalar_rasters


def rake(
    census_data: gpd.GeoDataFrame,
    raw_gdf: gpd.GeoDataFrame,
    raw_prediction: rt.RasterArray,
) -> rt.RasterArray:
    overlay_gdf = (
        census_data
        .overlay(
            raw_gdf,
            how="intersection",
            keep_geom_type=True,
        )
    )

    overlay_gdf["isection_area"] = overlay_gdf.area

    overlay_gdf["pixel_coverage"] = safe_divide(
        overlay_gdf["isection_area"],
        overlay_gdf["pixel_area"],
    )
    overlay_gdf["covered_pixel_population"] = overlay_gdf["pixel_population"] * overlay_gdf["pixel_coverage"]

    overlay_gdf["shape_pixel_population"] = overlay_gdf.groupby("shape_id")["covered_pixel_population"].transform(sum)
    overlay_gdf["raking_factor"] = safe_divide(
        overlay_gdf["population_total"].astype(np.float32),
        overlay_gdf["shape_pixel_population"],
    )
    overlay_gdf["raked_pixel_population"] = overlay_gdf["covered_pixel_population"] * overlay_gdf["raking_factor"]

    idx = np.arange(raw_prediction.size)
    raked_population = (
        overlay_gdf.groupby("pixel_id")["raked_pixel_population"]
        .sum()
        .sort_index()
        .reindex(idx, fill_value=np.nan)
        .to_numpy()
        .astype(np.float32)
        .reshape(raw_prediction.shape)
    )
    raked_raster = rt.RasterArray(
        data=raked_population,
        transform=raw_prediction.transform,
        crs=raw_prediction.crs,
        no_data_value=np.nan,
    )

    return raked_raster
