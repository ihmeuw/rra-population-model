from collections.abc import Iterator
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
from affine import Affine
from rasterio import features
from shapely import area, box, intersection, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.ops import unary_union

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

STEP_LIMIT = 1.0025  # 1% per year

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

    drop_iso3s = [
    ]
    keep_years = np.unique([
        int(year.split("q")[0]) for year in pmc.MODELING_TIME_POINTS
    ]).tolist()
    data_years = [
        i for i in available_census_years
        if int(i[1]) in keep_years
        and i[0] not in drop_iso3s
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
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, rt.RasterArray]:
    """Build the census/pixel overlay skeleton and per-pixel prediction table.

    Assumes the prediction nodata footprint (the water mask) is time-invariant
    across ``model_time_points``. That mask is produced upstream in the
    building-density pipeline (a separate codebase), where it is a single static
    raster applied to every time point, so the assumption holds today. Two places
    here rely on it, and both would be wrong if it ever became time-varying:

      * the no-population branch builds the saved footprint from
        ``model_time_points[0]`` alone (and census_rake_main reuses it for every
        time point);
      * the populated branch returns the last time point's raster as the
        ``template_raster``, whose valid mask ``build_overlay`` uses to decide
        which pixels to keep for *all* time points. A pixel that were valid at
        one time point but nodata at another would be mis-handled.

    If the mask ever varies by time point, this stage must derive a per-time-point
    valid mask instead.
    """
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

    overlay_skeleton: pd.DataFrame | None = None
    prediction_data: pd.DataFrame | None = None
    if task_map["population_total"] == 0:
        # Footprint from one time point only; relies on the time-invariant water
        # mask documented in this function's docstring.
        prediction_raster = rt.merge([
            pm_data.load_raw_prediction(block_key, model_time_points[0], model_spec)
            for block_key in block_keys
        ])
        prediction_raster = prediction_raster.clip(task_map["geometry"]).mask(task_map["geometry"])
        prediction_array = prediction_raster.to_numpy()
        no_data_value = prediction_raster.no_data_value
        if not np.isnan(no_data_value):
            raise ValueError("Unexpected no_data_value")
        valid_mask = ~np.isnan(prediction_array)
        if not valid_mask.any():
            # No predicted pixels in the admin (e.g. an all-water maritime/Antarctic
            # claim). Write a minimal 1x1 all-nodata raster instead of the full box:
            # the output tif still exists -- so a *missing* tif unambiguously means a
            # failed job, not an empty admin -- but it's tiny, and the rake stage
            # drops all-nodata rasters from its merge so it contributes nothing.
            template_raster = rt.RasterArray(
                np.full((1, 1), no_data_value, dtype=prediction_array.dtype),
                transform=prediction_raster.transform,
                crs=prediction_raster.crs,
                no_data_value=no_data_value,
            )
        else:
            template_raster = rt.RasterArray(
                np.where(valid_mask, 0, no_data_value).astype(prediction_array.dtype),
                transform=prediction_raster.transform,
                crs=prediction_raster.crs,
                no_data_value=no_data_value,
            )
    else:
        # Read only the task parent's bounding box, not the whole country. The
        # parent's most-detailed children are all inside the parent geometry (so
        # inside its bbox), and task_admins is in the same CRS as the census
        # parquet (ESRI:54034), so the parent geometry is a valid bbox. This is
        # the dominant per-task memory cost (e.g. the full USA census is ~40 GB).
        census_data = pm_data.load_census_data(
            task_map["iso3"],
            task_map["census_time_point"].split("q")[0],
            bounds=task_map["geometry"],
        )
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

        # Only the covered pixels (interior + border of the census shapes, minus
        # nodata) are ever used downstream. Compute them once from the overlay
        # skeleton (built from the first time point) and keep populations for just
        # those pixels, keyed by the raster's row-major index (pixel_id). This
        # keeps peak memory ~ covered_pixels x n_time_points instead of
        # box_pixels x n_time_points -- a win for admins whose bounding box is
        # mostly nodata (e.g. lots of ocean). Relies on the time-invariant water
        # mask documented above (covered pixels are fixed across time points).
        def load_prediction(model_time_point: str) -> rt.RasterArray:
            raster = rt.merge([
                pm_data.load_raw_prediction(block_key, model_time_point, model_spec)
                for block_key in block_keys
            ])
            return raster.clip(buffered_parent_geometry).mask(buffered_parent_geometry)

        template_raster = load_prediction(model_time_points[0])
        # Build the census/pixel overlay skeleton once and return it, so rake reuses
        # it instead of recomputing -- the border overlay is the expensive step on
        # convoluted admins, and it otherwise runs twice (here and in rake).
        overlay_skeleton = build_overlay_skeleton(census_data, template_raster)
        covered_pixel_ids = np.unique(overlay_skeleton["pixel_id"].to_numpy())
        pixel_populations = {}
        for model_time_point in model_time_points:
            prediction_raster = (
                template_raster
                if model_time_point == model_time_points[0]
                else load_prediction(model_time_point)
            )
            pixel_populations[f"pixel_population_{model_time_point}"] = (
                prediction_raster.to_numpy().flatten()[covered_pixel_ids]
            )
        prediction_data = pd.DataFrame(pixel_populations, index=covered_pixel_ids)
        prediction_data.index.name = "pixel_id"
        prediction_data = prediction_data.reset_index()

    return overlay_skeleton, prediction_data, template_raster


def calculate_time_point_raking_factor(
    shapes: pd.DataFrame,
    sorted_model_time_points: list[str],
    step: int,
    rf_anchor: npt.NDArray[np.floating[Any]],
) -> tuple[pd.DataFrame, npt.NDArray[np.floating[Any]]]:
    """Set ``raking_factor_<model_time_point>`` on the per-shape ``shapes`` table.

    Pure per-shape math (one row per census shape): the step-limit logic only ever
    uses shape_population / raking_factor / population_total, which are constant
    within a shape. The caller applies the resulting factor per pixel.
    """
    model_time_point = sorted_model_time_points[step]
    prev_model_time_point = sorted_model_time_points[step - 1]

    # Find the closest non-zero previous time point as the step-limit reference.
    # A zero shape_population reference collapses raked_pop_prev to zero, which
    # would force raked_pop_t to zero even after predictions recover.
    shape_population_prev = shapes[f"shape_population_{prev_model_time_point}"].to_numpy(copy=True)
    raking_factor_prev = shapes[f"raking_factor_{prev_model_time_point}"].to_numpy(copy=True)
    for earlier_tp in reversed(sorted_model_time_points[:step - 1]):
        still_zero = (shape_population_prev == 0) & (shapes[f"shape_population_{model_time_point}"].to_numpy() != 0)
        if not still_zero.any():
            break
        shape_population_prev[still_zero] = shapes.loc[still_zero, f"shape_population_{earlier_tp}"].to_numpy(copy=True)
        raking_factor_prev[still_zero] = shapes.loc[still_zero, f"raking_factor_{earlier_tp}"].to_numpy(copy=True)

    # Step-limit bounds on raking factor:
    # raked_pop_t = shape_pop_t * rf_t must stay in [raked_pop_prev/STEP_LIMIT, raked_pop_prev*STEP_LIMIT]
    # where raked_pop_prev = shape_pop_prev * rf_prev (from the closest non-zero reference).
    prev_mask = np.isfinite(raking_factor_prev)
    raked_pop_prev = np.full_like(shape_population_prev, np.nan)
    raked_pop_prev[prev_mask] = shape_population_prev[prev_mask] * raking_factor_prev[prev_mask]

    # Only apply bounds when both the reference and the current prediction are valid.
    shape_pop_current = shapes[f"shape_population_{model_time_point}"].to_numpy()
    valid_ref = (shape_pop_current > 0) & np.isfinite(raked_pop_prev)
    rf_lower = np.full_like(raking_factor_prev, 0.0)
    rf_lower[valid_ref] = safe_divide(
        raked_pop_prev[valid_ref],
        shape_pop_current[valid_ref] * STEP_LIMIT
    )
    rf_upper = np.full_like(raking_factor_prev, np.inf)
    rf_upper[valid_ref] = safe_divide(
        raked_pop_prev[valid_ref] * STEP_LIMIT,
        shape_pop_current[valid_ref]
    )
    # Absolute cap: raked_pop_t <= step * STEP_LIMIT * census_population.
    # Prevents unbounded compounding over many steps.
    census_pop = shapes["population_total"].astype(np.float32).to_numpy()
    rf_upper_abs = np.full_like(raking_factor_prev, np.inf)
    rf_upper_abs[valid_ref] = safe_divide(
        census_pop[valid_ref] * (step * STEP_LIMIT),
        shape_pop_current[valid_ref]
    )
    rf_upper = np.minimum(rf_upper, rf_upper_abs)

    # Target: rf_anchor, the first finite raking factor encountered stepping away from
    # the census time point (passed in and maintained by the caller). Each step aims for
    # this value so the raking factor converges back to the census-consistent level after
    # large model jumps instead of staying permanently offset.
    # Where rf_anchor is still inf/nan (no finite step seen yet), fall back to
    # raking_factor_direct so zero-to-nonzero transitions are treated independently.
    valid_direct = shape_pop_current > 0
    raking_factor_direct = np.full_like(raking_factor_prev, np.inf)
    raking_factor_direct[valid_direct] = safe_divide(
        shapes["population_total"].astype(np.float32).to_numpy()[valid_direct],
        shape_pop_current[valid_direct],
    )
    valid_anchor = np.isfinite(rf_anchor)
    rf_target = np.where(valid_anchor, rf_anchor, raking_factor_direct)
    shapes[f"raking_factor_{model_time_point}"] = np.clip(rf_target, rf_lower, rf_upper)

    # Replace any remaining inf/nan (shape_pop=0 at this step) with raking_factor_direct,
    # which will itself be inf when shape_pop=0 — those get zeroed out later.
    invalid_rf = ~np.isfinite(shapes[f"raking_factor_{model_time_point}"])
    shapes.loc[invalid_rf, f"raking_factor_{model_time_point}"] = raking_factor_direct[invalid_rf]

    # Update anchor: fill in shapes where we just computed the first finite raking factor.
    new_rf = shapes[f"raking_factor_{model_time_point}"].to_numpy()
    rf_anchor = np.where(~np.isfinite(rf_anchor) & np.isfinite(new_rf), new_rf, rf_anchor)

    return shapes, rf_anchor


def compute_shape_raking_factors(
    shape_id: npt.NDArray[np.integer[Any]],
    coverage: npt.NDArray[np.floating[Any]],
    population_total: npt.NDArray[np.floating[Any]],
    prediction_data: pd.DataFrame,
    pred_row: npt.NDArray[np.integer[Any]],
    census_time_point: str,
    model_time_points: list[str],
) -> pd.DataFrame:
    """Per-shape raking factor for each model time point.

    Reduces the (covered-pixel) overlay to one row per census shape, runs the
    step-limit raking there, and returns a table indexed by the integer shape code
    with a ``raking_factor_<model_time_point>`` column per time point. The caller maps
    these back onto pixels. This keeps the raking working set at n_shapes x n_tps
    instead of covered_pixels x (~5 x n_tps).

    ``shape_id``, ``coverage`` and ``population_total`` are per-(covered-pixel-row)
    arrays; ``pred_row`` maps each of those rows to its row in ``prediction_data`` so
    the per-time-point pixel populations can be gathered without a merge.
    """
    # per-shape census population (constant within a shape) sets the shape order/index
    shapes = pd.Series(population_total).groupby(shape_id).first().to_frame("population_total")
    for model_time_point in model_time_points:
        pixel_population = prediction_data[f"pixel_population_{model_time_point}"].to_numpy()[pred_row]
        covered = pixel_population * coverage
        shapes[f"shape_population_{model_time_point}"] = (
            pd.Series(covered).groupby(shape_id).sum()
        )

    shapes[f"raking_factor_{census_time_point}"] = safe_divide(
        shapes["population_total"].astype(np.float32),
        shapes[f"shape_population_{census_time_point}"],
    )

    distances = [
        (int(census_time_point.split("q")[0]) + int(census_time_point.split("q")[1]) / 4)
        -
        (int(model_time_point.split("q")[0]) + int(model_time_point.split("q")[1]) / 4)
        for model_time_point in model_time_points
    ]
    pre_sorted_model_time_points = [i[1] for i in sorted(zip(distances, model_time_points)) if i[0] >= 0]
    post_sorted_model_time_points = [i[1] for i in sorted(zip(distances, model_time_points), reverse=True) if i[0] <= 0]
    # rf_anchor: per-shape target raking factor, initialized from the census time
    # point and updated to the first finite value encountered as we step away from it.
    rf_anchor_init = shapes[f"raking_factor_{census_time_point}"].to_numpy().copy()
    for sorted_model_time_points in [pre_sorted_model_time_points, post_sorted_model_time_points]:
        rf_anchor = rf_anchor_init.copy()
        for i in range(1, len(sorted_model_time_points)):
            shapes, rf_anchor = calculate_time_point_raking_factor(
                shapes,
                sorted_model_time_points,
                i,
                rf_anchor,
            )

    raking_factor_cols = [f"raking_factor_{model_time_point}" for model_time_point in model_time_points]
    for col in raking_factor_cols:
        shapes.loc[~np.isfinite(shapes[col]), col] = 0

    return shapes[raking_factor_cols]


def build_overlay_skeleton(
    census_data: gpd.GeoDataFrame,
    template_raster: rt.RasterArray,
) -> pd.DataFrame:
    """Map census shapes onto prediction pixels via rasterization (geometry only).

    Returns one row per (shape, pixel) intersection with its ``isection_area``,
    ``pixel_area`` and ``population_total`` -- everything except the per-time-point
    ``pixel_population`` columns, which ``build_overlay`` attaches afterwards.

    Interior pixels (no shape boundary passing through them) lie entirely within
    a single shape, so a rasterized center hit gives exact full coverage with no
    geometry work. Only border pixels (cost ~ perimeter, not area) are
    polygonized and exactly intersected, and a border pixel may split across
    several shapes.

    Nodata (e.g. water) pixels are dropped: they carry no prediction and would
    only contribute zero. This makes multi-admin tasks match the behavior the
    single-admin path already had (in-shape water -> NaN rather than 0).
    """
    out_shape = template_raster.shape
    transform = template_raster.transform
    crs = template_raster.crs
    pixel_area = np.float32(abs(template_raster.x_resolution * template_raster.y_resolution))

    valid = _get_valid_data_mask(
        template_raster.to_numpy().astype(np.float32),
        template_raster.no_data_value,
    )

    census = census_data.reset_index(drop=True)
    population_total = census["population_total"].to_numpy()
    n_shapes = len(census)
    id_dtype = "uint16" if n_shapes < np.iinfo(np.uint16).max else "uint32"

    # Burn shape ids (1-based; 0 == no shape) by pixel center, then drop water.
    assigned = features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(census.geometry.to_numpy())),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=id_dtype,
    )
    assigned[~valid] = 0

    # Any pixel a shape boundary touches needs an exact intersection.
    border_mask = features.rasterize(
        ((geom, 1) for geom in census.geometry.boundary.to_numpy()),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    border_mask &= valid

    # Interior pixels: full coverage, exactly one shape each, no geometry needed.
    # ``shape_id`` here is the 0-based row position into ``census`` (an integer *code*,
    # not the string admin id). The string id is never needed downstream -- pixels are
    # only grouped by shape and census totals looked up by shape -- and integer codes
    # make the per-shape groupby and the per-time-point factor map far cheaper than
    # hashing a string on every one of the (up to ~150M) rows.
    interior_idx = np.flatnonzero((assigned > 0) & ~border_mask)
    interior = pd.DataFrame({
        "pixel_id": interior_idx,
        "shape_id": (assigned.ravel()[interior_idx] - 1).astype(id_dtype),
        "isection_area": pixel_area,
    })

    # Border pixels: build cell polygons only here and intersect with the shapes.
    border_idx = np.flatnonzero(border_mask)
    if border_idx.size:
        rows, cols = np.unravel_index(border_idx, out_shape)
        a, b, c, d, e, f = (
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        )
        x0 = a * cols + b * rows + c
        y0 = d * cols + e * rows + f
        x1 = a * (cols + 1) + b * (rows + 1) + c
        y1 = d * (cols + 1) + e * (rows + 1) + f
        boxes = box(
            np.minimum(x0, x1), np.minimum(y0, y1),
            np.maximum(x0, x1), np.maximum(y0, y1),
        )
        border_pixels = gpd.GeoDataFrame(
            {"pixel_id": border_idx}, geometry=boxes, crs=crs
        )
        # Pair border pixels with the shapes they hit (sjoin), then intersect
        # vectorized and take the area. This avoids geopandas.overlay's noding/
        # polygonize machinery, which is far slower on convoluted boundaries;
        # zero-area (line/point) intersections drop out via the area > 0 filter.
        shapes = census[["shape_id", "geometry"]].reset_index(drop=True)
        joined = gpd.sjoin(border_pixels, shapes, predicate="intersects", how="inner")
        shape_geom = shapes.geometry.to_numpy()[joined["index_right"].to_numpy()]
        isection_area = area(intersection(joined.geometry.to_numpy(), shape_geom))
        # index_right is the positional index into ``shapes`` (== census row), i.e. the
        # same integer shape code the interior pixels carry.
        border = pd.DataFrame({
            "pixel_id": joined["pixel_id"].to_numpy(),
            "shape_id": joined["index_right"].to_numpy().astype(id_dtype),
            "isection_area": isection_area.astype(np.float32),
        })
        border = border[border["isection_area"] > 0]
    else:
        border = pd.DataFrame({
            "pixel_id": np.array([], dtype=np.int64),
            "shape_id": np.array([], dtype=id_dtype),
            "isection_area": np.array([], dtype=np.float32),
        })

    overlay = pd.concat([interior, border], ignore_index=True)
    overlay["pixel_area"] = pixel_area
    # Census total per shape by integer-indexing the code (was a string merge on shape_id).
    overlay["population_total"] = population_total[overlay["shape_id"].to_numpy()]

    return overlay


def build_overlay(
    census_data: gpd.GeoDataFrame,
    prediction_data: pd.DataFrame,
    template_raster: rt.RasterArray,
) -> pd.DataFrame:
    """Attach per-time-point pixel populations to the census/pixel overlay skeleton.

    ``prediction_data`` only needs rows for the covered pixels (interior + border);
    ``process_census_data`` already restricts it to those, which is what keeps peak
    memory off the full bounding box.
    """
    overlay = build_overlay_skeleton(census_data, template_raster)
    pop_cols = [c for c in prediction_data.columns if c.startswith("pixel_population_")]
    overlay = overlay.merge(prediction_data[["pixel_id", *pop_cols]], on="pixel_id", how="left")
    return overlay


def rake(
    overlay_skeleton: pd.DataFrame,
    prediction_data: pd.DataFrame,
    template_raster: rt.RasterArray,
    census_time_point: str,
    model_time_points: list[str],
) -> Iterator[rt.RasterArray]:
    """Yield one raked raster per model time point, in ``model_time_points`` order.

    Takes the overlay skeleton already built by ``process_census_data`` (so the
    expensive border overlay isn't recomputed) and attaches the per-time-point
    pixel populations to it here.

    A generator (not a list) so the caller can save and free each raster before
    the next is built: peak output memory is a single box-sized raster instead of
    ``n_time_points`` of them. The per-pixel raked values are summed once (over the
    covered pixels only); only the final reshape to the full raster grid is done
    one time point at a time.
    """
    shape_id = overlay_skeleton["shape_id"].to_numpy()
    population_total = overlay_skeleton["population_total"].to_numpy()
    coverage = safe_divide(
        overlay_skeleton["isection_area"].to_numpy(),
        overlay_skeleton["pixel_area"].to_numpy(),
    ).clip(0, 1)

    # Align each skeleton row to its prediction row without a merge. prediction_data
    # is keyed by covered pixel_id and built from np.unique(...), so its pixel_id
    # column is sorted and contains every skeleton pixel_id; searchsorted gives the
    # exact prediction row for each skeleton row -- the same values a left merge on
    # pixel_id would -- without ever materializing the n_tps population columns onto
    # the (up to ~150M-row) overlay. That duplicate of the prediction table plus the
    # join transient was the dominant rake memory cost.
    skel_pixel = overlay_skeleton["pixel_id"].to_numpy()
    pred_pixel = prediction_data["pixel_id"].to_numpy()
    pred_row = np.searchsorted(pred_pixel, skel_pixel)

    # Raking factors are computed on a small per-shape table; the per-pixel apply
    # (covered_population * raking_factor -> sum per pixel) is done one time point at
    # a time below so no covered_pixels x n_tps intermediate is ever materialized.
    shape_raking_factors = compute_shape_raking_factors(
        shape_id, coverage, population_total, prediction_data, pred_row,
        census_time_point, model_time_points,
    )

    shape_id_series = pd.Series(shape_id)
    # Factorize pixel_id once (values are flattened raster positions). Per time
    # point we then sum-per-pixel with np.bincount and scatter into the box grid --
    # avoiding a groupby (re-hash) and a full-box reindex on every time point, which
    # is what made large admins slow.
    codes, covered_positions = pd.factorize(skel_pixel, sort=True)
    n_covered = len(covered_positions)
    size = template_raster.size
    for model_time_point in model_time_points:
        raking_factor = shape_id_series.map(
            shape_raking_factors[f"raking_factor_{model_time_point}"]
        ).to_numpy()
        pixel_population = prediction_data[f"pixel_population_{model_time_point}"].to_numpy()[pred_row]
        raked_pixel_population = pixel_population * coverage * raking_factor
        per_pixel = np.bincount(
            codes,
            weights=np.nan_to_num(raked_pixel_population),
            minlength=n_covered,
        )
        raked_population = np.full(size, np.nan, dtype=np.float32)
        raked_population[covered_positions] = per_pixel
        raked_data = raked_population.reshape(template_raster.shape)
        raked_raster = rt.RasterArray(
            data=raked_data,
            transform=template_raster.transform,
            crs=template_raster.crs,
            no_data_value=np.nan,
        )
        yield trim_null_edges(raked_raster)
