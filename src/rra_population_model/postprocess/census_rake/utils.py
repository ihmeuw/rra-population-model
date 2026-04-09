from typing import Any

from affine import Affine
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
from shapely import set_precision
from shapely.geometry import MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

STEP_LIMIT = 3

ADMIN_EXCLUSIONS = {
    "ARG": [
        "94028",  # Antarctica
    ],
}


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
    census_weights["distance"] = census_weights.groupby(["iso3", "model_time_point"])["distance"].transform("sum") - census_weights["distance"]
    census_weights["weight"] = (census_weights["distance"] / census_weights.groupby(["iso3", "model_time_point"])["distance"].transform("sum")).fillna(1)
    census_weights = (
        census_weights
        .drop("distance", axis=1)
        .set_index("census_time_point", append=True)
        .sort_index()
    )

    return census_weights


def get_task_admins(admins: gpd.GeoDataFrame, task_level: int) -> gpd.GeoDataFrame:
    parent_admins = admins.loc[admins["admin_level"] == task_level]
    parent_admins = parent_admins.rename(
        columns={
            "shape_id": "task_parent_id",
            "admin_level": "task_parent_level",
        },
    )
    parent_admins["area"] = parent_admins.area
    bounds = parent_admins.bounds
    parent_admins["bounds_area"] = (bounds['maxx'] - bounds['minx']) * (bounds['maxy'] - bounds['miny'])
    parent_admins["perimeter"] = parent_admins.geometry.length
    parent_admins = (
        parent_admins
        .set_index(["iso3", "census_time_point", "task_parent_id", "task_parent_level"])
        .loc[:, ["geometry", "area", "bounds_area", "perimeter", "population_total"]]
    )

    admins = admins.loc[admins["admin_level"] == admins["admin_level"].max()]
    admins["task_parent_id"] = admins["path_to_top_parent"].str.split(",").str[task_level]
    admins["task_parent_level"] = task_level
    admins = admins.loc[:, ["iso3", "census_time_point", "task_parent_id", "task_parent_level"]].value_counts().rename("most_detailed_units")
    admins = parent_admins.join(admins)

    return admins


def generate_census_inputs(
    pm_data: PopulationModelData,
    start_level: int,
    bounds_area_threshold: float,
    area_no_pop_threshold: float,
    admins_threshold: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    available_census_years = pm_data.list_census_data()

    keep_iso3s = [
        "AUS", "ARG", "BRA", "CAN", "CZE", "ESP", "GRC", "ISL", 
        "MDV", "MEX", "MLT", "MYS", "NPL", "PAN", "POL", "PRT",
        "QAT", "ROU", "RWA", "SVK", "TLS", "TON", "TZA", "VUT",
        "USA", "ZAF",
    ]
    data_years = [
        i for i in available_census_years
        if i[0] in keep_iso3s
    ]

    task_admins = []
    census_years = []
    for iso3, year, quarter in data_years:
        print(f"Processing {iso3} {year}q{quarter}")
        admins = pm_data.load_census_data(iso3, year)
        admins["population_total"] = admins["population_total"].fillna(0)
        admins["iso3"] = iso3
        admins["census_time_point"] = f"{year}q{quarter}"

        task_level = min(start_level, admins["admin_level"].max())
        admins = admins.loc[admins["admin_level"] >= task_level]
        task_level_admins = get_task_admins(admins.copy(), task_level)
        opt_check = task_level < admins["admin_level"].max()
        while opt_check:
            bounds_area_exceed = task_level_admins["bounds_area"] > bounds_area_threshold
            admins_exceed = task_level_admins["most_detailed_units"] > admins_threshold
            has_population = task_level_admins["population_total"] > 0
            area_no_pop_exceed = task_level_admins["area"] > area_no_pop_threshold
            to_split = (bounds_area_exceed | admins_exceed) & (has_population | area_no_pop_exceed)
            if to_split.any():
                reduce_parents = (
                    task_level_admins
                    .loc[to_split]
                    .index
                    .get_level_values("task_parent_id")
                    .tolist()
                )
                task_level_admins = task_level_admins.drop(reduce_parents, level="task_parent_id", errors="ignore")
                admins[f"level_{task_level}_id"] = admins["path_to_top_parent"].str.split(",").str[task_level]
                task_level_admins_supp = get_task_admins(
                    admins.loc[admins[f"level_{task_level}_id"].isin(reduce_parents)].copy(),
                    task_level + 1,
                )
                task_level_admins = pd.concat([task_level_admins, task_level_admins_supp])
                task_level += 1
                opt_check = task_level < admins["admin_level"].max()
            else:
                opt_check = False

        task_level_admins = task_level_admins.drop(ADMIN_EXCLUSIONS.get(iso3, []), level="task_parent_id", errors="ignore")
        task_admins.append(task_level_admins)
        census_years.append(
            pd.DataFrame({"census_time_point": f"{year}q{quarter}"}, index=pd.Index([iso3], name="iso3"))
        )
    task_admins = pd.concat(task_admins)
    census_years = pd.concat(census_years)

    census_weights = calculate_tp_weights(census_years)

    return task_admins, census_weights


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


def geometry_collection_to_multipolygon(geom: GeometryCollection | MultiPolygon) -> None | MultiPolygon:
    if isinstance(geom, GeometryCollection) and not isinstance(geom, MultiPolygon):
        polygons = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if polygons:
            return unary_union(polygons)
        return None
    return geom


def process_census_data(
    resolution: str,
    version: str,
    pm_data: PopulationModelData,
    task_map: gpd.GeoSeries,
    model_time_points: list[str],
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None, rt.RasterArray]:
    modeling_frame = pm_data.load_modeling_frame(resolution)
    model_spec = pm_data.load_model_specification(resolution, version)
    buffer_size = np.ceil(np.sqrt(int(resolution) ** 2 / 2))

    buffered_parent_geometry = task_map["geometry"].buffer(buffer_size)
    block_keys = (
        modeling_frame.
        loc[modeling_frame.intersects(buffered_parent_geometry), "block_key"]
        .unique()
        .tolist()
    )

    census_data: pd.DataFrame | None = None
    prediction_data: pd.DataFrame | None = None
    if task_map["population_total"] == 0:
        prediction_raster = rt.merge([
            pm_data.load_raw_prediction(block_key, model_time_points[0], model_spec)
            for block_key in block_keys
        ])
        prediction_raster = prediction_raster.clip(task_map["geometry"]).mask(task_map["geometry"])
        prediction_array = prediction_raster.to_numpy()
        no_data_value = prediction_raster.no_data_value
        if not np.isnan(no_data_value):
            raise ValueError("Unexpected no_data_value")
        prediction_raster = rt.RasterArray(
            np.where(
                np.isnan(prediction_array), no_data_value, 0
            )
            .astype(prediction_array.dtype),
            transform=prediction_raster.transform,
            crs=prediction_raster.crs,
            no_data_value=no_data_value,
        )
    else:
        census_data = pm_data.load_census_data(task_map["iso3"], task_map["census_time_point"].split("q")[0])
        census_data = census_data.loc[
            census_data["admin_level"] == census_data["admin_level"].max()
        ]
        census_data = census_data.loc[
            census_data["path_to_top_parent"].apply(lambda x: x.split(",")[task_map["task_parent_level"]] == task_map["task_parent_id"])
        ]
        census_data = census_data.loc[:, ["shape_id", "population_total", "geometry"]]
        census_data = census_data.to_crs(modeling_frame.crs)

        invalid_admins = ~census_data["geometry"].is_valid
        if invalid_admins.any():
            census_data.loc[invalid_admins, "geometry"] = census_data.loc[invalid_admins, "geometry"].make_valid()
        if census_data["geometry"].apply(lambda x: isinstance(x, GeometryCollection)).any():
            census_data["geometry"] = census_data["geometry"].apply(geometry_collection_to_multipolygon)
        if not census_data["geometry"].is_valid.all():
            raise ValueError("Invalid admins remain")
        census_data["geometry"] = census_data["geometry"].map(lambda g: set_precision(g, 0.01))

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
        prediction_data = prediction_data.reset_index()
        # prediction_data["geometry"] = prediction_data.geometry.map(lambda g: set_precision(g, 0.01))
        prediction_data["pixel_area"] = prediction_data.area

    return census_data, prediction_data, prediction_raster


def generate_and_apply_raking_factors(
    overlay_gdf: pd.DataFrame,
    census_time_point: str,
    model_time_points: list[str],
) -> pd.DataFrame:
    for model_time_point in model_time_points:
        overlay_gdf[f"covered_pixel_population_{model_time_point}"] = overlay_gdf[f"pixel_population_{model_time_point}"] * overlay_gdf["pixel_coverage"]
        overlay_gdf[f"shape_population_{model_time_point}"] = overlay_gdf.groupby("shape_id")[f"covered_pixel_population_{model_time_point}"].transform("sum")

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
            if decrease.all():
                overlay_gdf[f"raking_factor_{model_time_point}"] = (
                    overlay_gdf.loc[:, ["raking_factor_decrease", f"raking_factor_{prev_model_time_point}"]].max(axis=1)
                )
            elif not decrease.any():
                overlay_gdf[f"raking_factor_{model_time_point}"] = (
                    overlay_gdf.loc[:, ["raking_factor_increase", f"raking_factor_{prev_model_time_point}"]].min(axis=1)
                )
            else:
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
        overlay_gdf[f"raked_pixel_population_{model_time_point}"] = overlay_gdf[f"covered_pixel_population_{model_time_point}"] * overlay_gdf[f"raking_factor_{model_time_point}"]

    spatial_cols = ["shape_id", "pixel_id", "isection_area", "pixel_area", "pixel_coverage"]
    raked_pixel_cols = [f"raked_pixel_population_{model_time_point}" for model_time_point in model_time_points]
    overlay_gdf = overlay_gdf.loc[:, spatial_cols + raked_pixel_cols]

    return overlay_gdf


def single_admin_overlay(
    census_gdf: gpd.GeoDataFrame,
    pixel_gdf: gpd.GeoDataFrame,
    template_raster: rt.RasterArray,
) -> pd.DataFrame:
    geometry = census_gdf.geometry.item()

    # Infer square pixel dimensions from area.
    pixel_area = float(pixel_gdf["pixel_area"].iloc[0])
    buffer_size = np.ceil(np.sqrt(pixel_area / 2))
    interior = geometry.buffer(-buffer_size)
    if not interior.is_valid:
        interior = interior.buffer(0)

    if interior.is_empty:
        interior_valid = np.zeros(template_raster.shape, dtype=bool)
    else:
        interior_mask_raster = template_raster.mask(gpd.GeoSeries([interior], crs=census_gdf.crs))
        interior_valid = _get_valid_data_mask(
            interior_mask_raster.to_numpy().astype(np.float32),
            interior_mask_raster.no_data_value
        )

    covered_area = np.zeros(len(pixel_gdf), dtype=np.float32)

    interior_pixel_ids = np.flatnonzero(interior_valid.ravel())
    full_rows = pixel_gdf["pixel_id"].isin(interior_pixel_ids)

    covered_area[full_rows.to_numpy()] = pixel_area

    # Use geometry predicates for border candidates to avoid mask discretization gaps.
    try:
        candidate_rows_arr = np.zeros(len(pixel_gdf), dtype=bool)
        candidate_rows_arr[pixel_gdf.sindex.query(geometry, predicate="intersects")] = True
        border_rows = pd.Series(candidate_rows_arr, index=pixel_gdf.index) & ~full_rows
    except Exception:
        border_rows = pixel_gdf.intersects(geometry) & ~full_rows

    if border_rows.any():
        border_geom = pixel_gdf.loc[border_rows, "geometry"]
        border_geom = border_geom.intersection(geometry)
        covered_area[border_rows.to_numpy()] = border_geom.area.to_numpy()
        pixel_gdf.loc[border_rows, "geometry"] = border_geom

    pixel_gdf["isection_area"] = covered_area
    pixel_gdf["shape_id"] = census_gdf.shape_id.item()
    pixel_gdf["population_total"] = census_gdf.population_total.item()

    pixel_gdf = pixel_gdf.loc[pixel_gdf["isection_area"] > 0].drop("geometry", axis=1)

    return pixel_gdf


def rake(
    census_data: gpd.GeoDataFrame,
    prediction_data: gpd.GeoDataFrame,
    template_raster: rt.RasterArray,
    census_time_point: str,
    model_time_points: list[str],
) -> list[rt.RasterArray]:
    if len(census_data) > 1:
        overlay_gdf = (
            census_data
            .overlay(
                prediction_data,
                how="intersection",
                keep_geom_type=True,
            )
        )
        overlay_gdf["isection_area"] = overlay_gdf.area
    else:
        overlay_gdf = single_admin_overlay(
            census_data.copy(),
            prediction_data.copy(),
            template_raster,
        )
    overlay_gdf["pixel_coverage"] = safe_divide(
        overlay_gdf["isection_area"],
        overlay_gdf["pixel_area"],
    )

    overlay_gdf = generate_and_apply_raking_factors(
        overlay_gdf, census_time_point, model_time_points
    )

    idx = np.arange(template_raster.size)
    overlay_gdf = (
        overlay_gdf
        .loc[:, ["pixel_id"] + [f"raked_pixel_population_{model_time_point}" for model_time_point in model_time_points]]
        .groupby("pixel_id")
        .sum()
        .sort_index()
        .reindex(idx, fill_value=np.nan)
    )
    raked_rasters = []
    for model_time_point in model_time_points:
        raked_population = (
            overlay_gdf[f"raked_pixel_population_{model_time_point}"]
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
