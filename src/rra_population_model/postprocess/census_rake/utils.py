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

STEP_LIMIT = 3


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
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, rt.RasterArray, list[rt.RasterArray]]:
    modeling_frame = pm_data.load_modeling_frame(resolution)
    model_spec = pm_data.load_model_specification(resolution, version)
    buffer_size = int(resolution) * 5

    census_data = pm_data.load_census_data(task_map["iso3"], census_time_point.split("q")[0])
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

    prediction_data: pd.DataFrame | None = None
    for model_time_point in model_time_points:
        prediction_raster = rt.merge([
            pm_data.load_raw_prediction(block_key, model_time_point, model_spec)
            for block_key in block_keys
        ])
        prediction_raster = prediction_raster.clip(buffered_parent_geometry).mask(buffered_parent_geometry)
        if prediction_data is None:
            prediction_data = (
                prediction_raster
                .to_gdf()
                .rename(columns={"value": f"pixel_population_{model_time_point}"})
            )
        else:
            prediction_data[f"pixel_population_{model_time_point}"] = prediction_raster.to_numpy().flatten()
    prediction_data.index.name = "pixel_id"
    prediction_data = prediction_data.sort_index().reset_index()
    prediction_data["geometry"] = prediction_data.geometry.map(lambda g: set_precision(g, 0.001))
    prediction_data["pixel_area"] = prediction_data.area

    return census_data, prediction_data, prediction_raster


def generate_raking_factors(
    overlay_gdf: gpd.GeoDataFrame,
    census_time_point: str,
    model_time_points: list[str],
) -> gpd.GeoDataFrame:
    for model_time_point in model_time_points:
        overlay_gdf[f"covered_pixel_population_{model_time_point}"] = overlay_gdf[f"pixel_population_{model_time_point}"] * overlay_gdf["pixel_coverage"]
        overlay_gdf[f"shape_population_{model_time_point}"] = overlay_gdf.groupby("shape_id")[f"covered_pixel_population_{model_time_point}"].transform(sum)

    overlay_gdf[f"raking_factor_{census_time_point}"] = safe_divide(
        overlay_gdf["population_total"].astype(np.float32),
        overlay_gdf[f"shape_population_{census_time_point}"],
    )

    distances = [
        (int(census_time_point.split("q")[0]) + int(census_time_point.split("q")[1]) / 4)
        -
        (int(model_time_point.split("q")[0]) + int(model_time_point.split("q")[1]) / 4)
        for model_time_point in model_time_points
    ]
    pre_sorted_model_time_points = [i[1] for i in sorted(zip(distances, model_time_points)) if i[0] >= 0]
    post_sorted_model_time_points = [i[1] for i in sorted(zip(distances, model_time_points), reverse=True) if i[0] <= 0]
    for sorted_model_time_points in [pre_sorted_model_time_points, post_sorted_model_time_points]:
        for i in range(1, len(sorted_model_time_points)):
            model_time_point = sorted_model_time_points[i]
            prev_model_time_point = sorted_model_time_points[i-1]

            # 1) calculate raking factors with constrained increase/decrease allowed from previous time step
            overlay_gdf["raking_factor_decrease"] = (
                overlay_gdf[f"shape_population_{prev_model_time_point}"]
                *
                (1 / (STEP_LIMIT * overlay_gdf[f"shape_population_{model_time_point}"]))
                *
                overlay_gdf[f"raking_factor_{prev_model_time_point}"]
            )
            overlay_gdf["raking_factor_decrease"] = overlay_gdf["raking_factor_decrease"].clip(-np.inf, 1)
            overlay_gdf["raking_factor_increase"] = (
                overlay_gdf[f"shape_population_{prev_model_time_point}"]
                *
                (STEP_LIMIT / overlay_gdf[f"shape_population_{model_time_point}"])
                *
                overlay_gdf[f"raking_factor_{prev_model_time_point}"]
            )
            overlay_gdf["raking_factor_increase"] = overlay_gdf["raking_factor_increase"].clip(1, np.inf)

            # 2) splice together constrained raking factors based on whether data time point is increasing or decreasing
            decrease = (
                overlay_gdf[f"shape_population_{model_time_point}"]
                <
                overlay_gdf[f"shape_population_{prev_model_time_point}"]
            )
            overlay_gdf[f"raking_factor_{model_time_point}"] = pd.concat([
                overlay_gdf.loc[decrease, ["raking_factor_decrease", f"raking_factor_{prev_model_time_point}"]].max(axis=1),
                overlay_gdf.loc[~decrease, ["raking_factor_increase", f"raking_factor_{prev_model_time_point}"]].min(axis=1),
            ]).sort_index()

            # 3) replace infs with previous time point OR try directly calculating if previous is also inf
            overlay_gdf.loc[
                np.isinf(overlay_gdf[f"raking_factor_{model_time_point}"]),
                f"raking_factor_{model_time_point}"
            ] = overlay_gdf[f"raking_factor_{prev_model_time_point}"]
            overlay_gdf["raking_factor_direct"] = safe_divide(
                overlay_gdf["population_total"].astype(np.float32),
                overlay_gdf[f"shape_population_{model_time_point}"],
            )
            overlay_gdf.loc[
                np.isinf(overlay_gdf[f"raking_factor_{model_time_point}"]),
                f"raking_factor_{model_time_point}"
            ] = overlay_gdf["raking_factor_direct"]

    for model_time_point in model_time_points:
        overlay_gdf.loc[
            np.isinf(overlay_gdf[f"raking_factor_{model_time_point}"]),
            f"raking_factor_{model_time_point}"
        ] = 0

    spatial_cols = ["shape_id", "pixel_id", "isection_area", "pixel_area", "pixel_coverage"]
    covered_pixel_cols = [f"covered_pixel_population_{model_time_point}" for model_time_point in model_time_points]
    raking_factor_cols = [f"raking_factor_{model_time_point}" for model_time_point in model_time_points]
    overlay_gdf = overlay_gdf.loc[:, spatial_cols + covered_pixel_cols + raking_factor_cols]

    return overlay_gdf


def rake(
    census_data: gpd.GeoDataFrame,
    prediction_data: gpd.GeoDataFrame,
    template_raster: rt.RasterArray,
    census_time_point: str,
    model_time_points: list[str],
) -> rt.RasterArray:
    overlay_gdf = (
        census_data
        .overlay(
            prediction_data,
            how="intersection",
            keep_geom_type=True,
        )
    )

    overlay_gdf["isection_area"] = overlay_gdf.area
    overlay_gdf["pixel_coverage"] = safe_divide(
        overlay_gdf["isection_area"],
        overlay_gdf["pixel_area"],
    )

    overlay_gdf = generate_raking_factors(
        overlay_gdf, census_time_point, model_time_points
    )

    idx = np.arange(template_raster.size)
    raked_rasters = []
    for model_time_point in model_time_points:
        overlay_gdf["raked_pixel_population"] = overlay_gdf[f"covered_pixel_population_{model_time_point}"] * overlay_gdf[f"raking_factor_{model_time_point}"]
        raked_population = (
            overlay_gdf.groupby("pixel_id")["raked_pixel_population"]
            .sum()
            .sort_index()
            .reindex(idx, fill_value=np.nan)
            .to_numpy()
            .astype(np.float32)
            .reshape(template_raster.shape)
        )
        raked_raster = rt.RasterArray(
            data=raked_population,
            transform=template_raster.transform,
            crs=template_raster.crs,
            no_data_value=np.nan,
        )
        raked_raster = trim_null_edges(raked_raster)
        raked_rasters.append(raked_raster)

    return raked_rasters
