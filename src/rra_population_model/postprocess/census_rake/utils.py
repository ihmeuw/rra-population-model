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

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

# v3-graded temporal mechanism constants. All three are measured, not chosen --
# derivations and evidence in .claude/census_rake_temporal/DESIGN.md:
#   K_PERSIST -- quarters a newly-appearing pixel must stay ON before its
#     population is creditable. Genuine construction accumulates and never
#     reverts; detector flicker reverts. Two independent measurements agree on
#     ~2: the separation point of on-run-length distributions for reverting vs
#     persisting pixels, and the buildings-precede-people occupancy lag.
#   ANCHOR_WINDOW -- quarters starting at the census time point in which an ON
#     pixel counts as existing stock; "new" requires OFF throughout the window.
#     Set by the measured flicker-misclassification vs missed-construction
#     curves (population-costed crossover at W~2-3 with phantom the costlier
#     error class) and brackets census reference dates (US April 1 = q1/q2).
#   RF_LAMBDA_PERSONS -- partial-pooling mass for the parent rate, denominated
#     in CENSUS PERSONS so it is invariant to the raw-prediction unit scale
#     (which shifts across model retrains -- e.g. the log-occupancy retrain
#     shrank units by ~two orders of magnitude; ~150x measured for USA/KOR):
#         RF_p = (C_p + LAMBDA) / (M_p + LAMBDA / RF_country)
#     i.e. the prior contributes LAMBDA people at the national rate. Below that
#     parent mass the parent's own rate estimate is noisier than the prior
#     (setting rule: split-half sampling noise = cross-parent dispersion).
#     800 persons restates the validated harness value (lambda = 100 model
#     units on the 2026_05_06 run, where national rates were 7.5 (USA) to
#     9.3 (KOR) persons/unit -> 750-930 persons).
K_PERSIST = 2
ANCHOR_WINDOW = 2
RF_LAMBDA_PERSONS = 800.0
# One-sided growth-density cap (adopted 2026-08-24 for the public release):
# a unit's implied mean people-per-ON-pixel may not exceed the q99 of its
# parent's ESTABLISHED family (census / on-pixels at the anchor, over
# well-measured units), unless the census itself put it higher --
# y <= max(C, q99 * n_on(t)). Never cuts below census (anchor exactness and
# at-anchor extremes untouched); only bounds construction-channel credit.
# Measured on the US audit (production semantics, 2026q1): caps 1,440 units,
# removing 1.32M of 11.1M credit (11.8%), concentrated in the density-flagged
# class whose predictions carry ~4.5x normal mass per new pixel (upstream
# building-layer artifact; the cap self-adjusts as upstream QC improves since
# the family is recomputed per run).
# Small families cannot estimate a tail quantile: subsampled q99 reads ~35%
# low at n=30 and ~21% low at n=100 (downsampling 250 large counties), so
# below DENSITY_CAP_MIN_UNITS the bound is DENSITY_CAP_SMALL_FAMILY_MULT x the
# family MAX instead -- a statistic that exists at any size and degrades
# gracefully to a self-cap for single-unit parents (which previously had no
# bound at all). k=3 is the ~p75 of the measured ratio q99_full/max(n) over
# the small-family size range (median 1.7-3.1 at n=3-15); measured impact of
# the whole hybrid rule vs the old (q99, min 30): identical on USA/KOR/ESP
# (zero additional caps) -- its reach is the single-unit-parent censuses.
DENSITY_CAP_QUANTILE = 0.99
DENSITY_CAP_MIN_UNITS = 100
DENSITY_CAP_SMALL_FAMILY_MULT = 3.0
DENSITY_CAP_MIN_CENSUS = 5.0

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


def process_census_data(
    resolution: str,
    version: str,
    pm_data: PopulationModelData,
    task_map: gpd.GeoSeries,
    model_time_points: list[str],
    pooling_level: int,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    rt.RasterArray,
    "pd.Series[Any] | None",
    "pd.Series[Any] | None",
]:
    """Build the census/pixel overlay skeleton and per-pixel prediction table.

    Also returns, in census row order (the order that defines the overlay
    skeleton's integer shape codes): the ``shape_id`` strings and each unit's
    semantic pooling-parent id (``path_to_top_parent`` element at
    ``pooling_level``), so callers can label per-shape outputs and look up
    pooled rates without re-reading the census.

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
    shape_ids: "pd.Series[Any] | None" = None
    pooling_ids: "pd.Series[Any] | None" = None
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
        # Row order defines the integer shape codes used everywhere downstream;
        # capture ids and semantic pooling parents before the column filter.
        pooling_ids = (
            census_data["path_to_top_parent"].str.split(",").str[pooling_level]
            .reset_index(drop=True)
        )
        census_data = census_data.loc[:, ["shape_id", "population_total", "geometry"]]
        census_data = census_data.to_crs(modeling_frame.crs)
        shape_ids = census_data["shape_id"].reset_index(drop=True)

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
            covered_values = prediction_raster.to_numpy().flatten()[covered_pixel_ids]
            if np.nanmin(covered_values) < 0:
                # Negative predictions would make w = built/(built+base) exceed
                # 1 without bound downstream; nothing upstream asserts this.
                msg = f"Negative raw predictions at {model_time_point}"
                raise ValueError(msg)
            pixel_populations[f"pixel_population_{model_time_point}"] = covered_values
        prediction_data = pd.DataFrame(pixel_populations, index=covered_pixel_ids)
        prediction_data.index.name = "pixel_id"
        prediction_data = prediction_data.reset_index()

    return overlay_skeleton, prediction_data, template_raster, shape_ids, pooling_ids


def margin_time_points(
    write_time_points: list[str],
    census_time_point: str,
    available_time_points: list[str],
) -> list[str]:
    """Time points the v3 pixel margins must be computed over.

    The written window (this census's weighted time points) extended with
    ``K_PERSIST`` quarters before its start (run-length warm-up, so a pixel
    already persisting when the window opens is credited from the first written
    time point) and enough after the census time point to complete the anchor
    window. Clipped to the time points that actually have predictions.
    """
    missing = set(write_time_points) - set(available_time_points)
    if missing:
        msg = (
            f"Written time points {sorted(missing)} have no raw predictions; "
            "refusing to build a margin window over missing inputs."
        )
        raise ValueError(msg)
    ordered = sorted(set(available_time_points) | set(write_time_points))
    first_i = ordered.index(sorted(write_time_points)[0])
    last_i = ordered.index(sorted(write_time_points)[-1])
    tc_i = ordered.index(census_time_point)
    lo = max(0, first_i - K_PERSIST)
    hi = min(len(ordered) - 1, max(last_i, tc_i + ANCHOR_WINDOW - 1))
    extended = [
        tp for tp in ordered[lo : hi + 1] if tp in set(available_time_points)
    ]
    return sorted(set(extended) | set(write_time_points))


def anchor_window(census_time_point: str, time_points: list[str]) -> list[str]:
    """The ANCHOR_WINDOW quarters that define existing stock for one census.

    Starts at the census time point; falls back to the preceding quarter when
    the census sits at the series end. ``time_points`` must be chronologically
    sorted and contain the census time point. Shared by compute_shape_values
    and the census_rf pre-stage so both classify stock — and measure the
    density-cap footprint — over identical quarters.
    """
    tc_i = time_points.index(census_time_point)
    window = time_points[tc_i : tc_i + ANCHOR_WINDOW]
    if len(window) < ANCHOR_WINDOW and tc_i > 0:
        window = time_points[max(0, tc_i - (ANCHOR_WINDOW - 1)) : tc_i + 1]
    return window


def compute_shape_values(
    shape_id: npt.NDArray[np.integer[Any]],
    coverage: npt.NDArray[np.floating[Any]],
    population_total: npt.NDArray[np.floating[Any]],
    prediction_data: pd.DataFrame,
    pred_row: npt.NDArray[np.integer[Any]],
    census_time_point: str,
    model_time_points: list[str],
    write_time_points: list[str],
    pooling_ids: "pd.Series[Any]",
    rf_by_parent: "pd.Series[Any]",
    bounds_by_parent: "pd.Series[Any]",
    rf_country: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-shape v3-graded populations and effective raking factors.

    The v3 mechanism ("flat + persistent-pixel construction channel", see
    .claude/census_rake_temporal/DESIGN.md), per census unit i with census C_i
    anchored at the census time point t_c:

        y_i(t)     = gate_i(t) * [ C_i + RF_p * built_i(t) * w_i(t) ]
        built_i(t) = population in pixels OFF throughout the anchor window
                     [t_c .. t_c+W-1] and ON >= K_PERSIST consecutive quarters
        w_i(t)     = built_i(t) / (built_i(t) + base_i),  base_i = pred_i(t_c)
        RF_p       = (C_p + LAMBDA) / (M_p(t_c) + LAMBDA / RF_country)
                     per SEMANTIC POOLING PARENT p (one admin level above the
                     census's most-detailed, capped at admin2; national when
                     admin1 is most detailed) -- precomputed by the census_rf
                     pre-stage and passed in as ``rf_by_parent``; units map to
                     parents via ``pooling_ids`` (census row order, from
                     path_to_top_parent). LAMBDA = RF_LAMBDA_PERSONS.
        gate_i(t)  = 1{unit has any covered pixel with population at t}

    The growth-density cap families use the SAME pooling parents (one "parent"
    for rate and bound); like RF_p, the per-parent bound is computed over the
    whole census by the pre-stage (``bounds_by_parent``) and looked up here,
    so task boundaries never truncate a family.

    Established stock is flat at census level (the model's per-unit growth
    signal has no out-of-sample skill); only persistent new construction earns
    credit, graded by how new the unit's stock is, and only FORWARD of the
    anchor (t >= t_c; earlier time points are flat at C), bounded by the
    one-sided growth-density cap (y <= max(C, q99_parent * n_on(t)); see the
    DENSITY_CAP_* constants). Exact at the anchor by algebra (built(t_c) = 0,
    and the cap never cuts below C). Census-0 units use the same channel.
    There is no step limit or per-unit rate division anywhere.

    ``model_time_points`` is the (chronological) margin window -- the written
    time points plus warm-up/anchor-window extensions -- over which pixel
    run-lengths are tracked; ``write_time_points`` are the time points that get
    factors/outputs. Returns ``(factors, shape_table)``: ``factors`` is indexed
    by the integer shape code with one finite float64
    ``raking_factor_<tp>`` column per written time point (y / predicted; 0 for
    ungated shapes) for the caller to map onto pixels -- the same contract the
    incumbent mechanism satisfied, so the pixel apply is unchanged and
    within-unit allocation follows the prediction at t. ``shape_table`` carries
    the per-shape quantities (census, base, built, y, on-pixel counts) for
    persistence.
    """
    write_set = set(write_time_points)
    # per-shape census population (constant within a shape) sets the shape order/index
    shapes = pd.Series(population_total).groupby(shape_id).first().to_frame("population_total")
    for model_time_point in model_time_points:
        pixel_population = prediction_data[f"pixel_population_{model_time_point}"].to_numpy()[pred_row]
        covered = pixel_population * coverage
        shapes[f"shape_population_{model_time_point}"] = (
            pd.Series(covered).groupby(shape_id).sum()
        )

    n_codes = int(shape_id.max()) + 1
    code_index = shapes.index.to_numpy()

    def by_shape(weights: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        summed: npt.NDArray[np.floating[Any]] = np.bincount(
            shape_id, weights=weights, minlength=n_codes
        )[code_index]
        return summed

    # Anchor-window classification: a pixel is existing stock if ON at any of
    # the anchor-window quarters (see anchor_window); "new" requires OFF
    # throughout the window.
    window = anchor_window(census_time_point, model_time_points)
    on_window = np.zeros(len(pred_row), dtype=bool)
    for tp in window:
        on_window |= (
            prediction_data[f"pixel_population_{tp}"].to_numpy()[pred_row] * coverage
        ) > 0
    new_px = ~on_window

    # K-consecutive-quarter persistence, tracked chronologically over the full
    # margin window (construction accumulates and never reverts; flicker does).
    run: npt.NDArray[np.int16] = np.zeros(len(pred_row), dtype=np.int16)
    built: dict[str, npt.NDArray[np.floating[Any]]] = {}
    n_on: dict[str, npt.NDArray[np.floating[Any]]] = {}
    for tp in model_time_points:
        covered_t = prediction_data[f"pixel_population_{tp}"].to_numpy()[pred_row] * coverage
        on_t = covered_t > 0
        run = np.where(on_t, run + 1, 0).astype(np.int16)
        if tp in write_set:
            persist = run >= K_PERSIST
            built[tp] = by_shape(np.where(new_px & persist, covered_t, 0.0))
            n_on[tp] = by_shape(on_t.astype(np.float64))

    census = shapes["population_total"].to_numpy(dtype=np.float64)
    base = shapes[f"shape_population_{census_time_point}"].to_numpy(dtype=np.float64)

    # Growth-density cap bound: precomputed per pooling parent by the census_rf
    # pre-stage over the WHOLE census (q99 of the parent's established density
    # family, k x family-max for small families; see the DENSITY_CAP_*
    # provenance comment and DESIGN section 6) and looked up here exactly like
    # RF_p. A parent with no well-measured units -- or missing from the table
    # -- is uncapped.
    parent_ids = pooling_ids.to_numpy()[code_index]
    bound_arr = (
        pd.Series(parent_ids).map(bounds_by_parent).fillna(np.inf)
        .to_numpy(dtype=np.float64)
    )

    # Per-unit pooled rate: look up the unit's semantic pooling parent in the
    # pre-stage table. A parent missing from the table (shouldn't happen --
    # same census file feeds both) falls back to the national rate.
    rf_arr = (
        pd.Series(parent_ids).map(rf_by_parent)
        .fillna(rf_country if rf_country > 0 else 0.0)
        .to_numpy(dtype=np.float64)
    )

    # Assemble both frames from column dicts in one construction (repeated
    # column insertion fragments the frame and warns on wide windows).
    factor_cols: dict[str, npt.NDArray[np.floating[Any]]] = {}
    table_cols: dict[str, npt.NDArray[Any]] = {
        "pooling_parent_id": parent_ids,
        "census_population": census,
        "base_population": base,
        "rf_parent": rf_arr,
        "density_bound": bound_arr,
    }
    for tp in write_time_points:
        predicted_t = shapes[f"shape_population_{tp}"].to_numpy(dtype=np.float64)
        built_t = built[tp]
        weight = np.where(built_t + base > 0, built_t / np.maximum(built_t + base, 1e-300), 0.0)
        # FORWARD-ONLY channel: credit applies at t >= t_c; before the anchor
        # the unit is flat at C. Pre-anchor "built" (stock present earlier but
        # gone by the anchor window) is dominated by detector reversals, and
        # crediting it degraded held-out accuracy everywhere it acted
        # (adversarial audit 2026-08-21, verified: ESP backcast touched-stratum
        # wMAPE 10.18->11.86; KOR +0.86/+0.32). It also removes the artificial
        # series-start credit step from run-length warm-up truncation. built_t
        # is still persisted in the shape table as a diagnostic.
        credit_t = rf_arr * built_t * weight if tp >= census_time_point else 0.0
        y_t = np.where(predicted_t > 0, census + credit_t, 0.0)
        # One-sided per-parent cap: bounds credit only; max(C, .) never cuts
        # below census, so the anchor and at-anchor extremes are untouched.
        finite_bound = np.isfinite(bound_arr)
        bounded_rate = np.where(finite_bound, bound_arr, 0.0)
        allowed = np.where(
            finite_bound, np.maximum(census, bounded_rate * n_on[tp]), np.inf
        )
        y_t = np.minimum(y_t, allowed)
        # Effective factor: exact reparameterization of y for the per-pixel
        # apply (pixel = pred * coverage * factor), so unit totals equal y and
        # within-unit allocation follows the prediction. Finite everywhere:
        # y > 0 requires predicted_t > 0 (the gate). float64 -- y over a tiny
        # prediction is large (it cancels in the apply) and must not saturate.
        factor_cols[f"raking_factor_{tp}"] = safe_divide(y_t, predicted_t)
        table_cols[f"shape_population_{tp}"] = predicted_t
        table_cols[f"built_{tp}"] = built_t
        table_cols[f"raked_{tp}"] = y_t
        table_cols[f"n_on_{tp}"] = n_on[tp].astype(np.int32)
    factors = pd.DataFrame(factor_cols, index=shapes.index)
    shape_table = pd.DataFrame(table_cols, index=shapes.index)

    # Exact at the anchor by algebra (built(t_c) = 0 by construction); a cheap
    # guard against regressions in the margin bookkeeping.
    gated = base > 0
    anchor_error = abs(
        shape_table[f"raked_{census_time_point}"].to_numpy()[gated].sum()
        - census[gated].sum()
    )
    if anchor_error > max(1e-6 * census[gated].sum(), 1e-6):
        msg = (
            f"v3 raked population is not exact at the census anchor "
            f"{census_time_point}: |error| = {anchor_error:.6g} people"
        )
        raise ValueError(msg)

    return factors, shape_table


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
    write_time_points: list[str],
    pooling_ids: "pd.Series[Any]",
    rf_by_parent: "pd.Series[Any]",
    bounds_by_parent: "pd.Series[Any]",
    rf_country: float,
) -> tuple[pd.DataFrame, Iterator[rt.RasterArray]]:
    """Per-shape v3 table plus one raked raster per written time point.

    Takes the overlay skeleton already built by ``process_census_data`` (so the
    expensive border overlay isn't recomputed). ``model_time_points`` is the
    margin window the mechanism needs (see ``margin_time_points``);
    ``write_time_points`` are the census's weighted time points that get
    rasters, yielded in that order.

    Rasters come as a generator (not a list) so the caller can save and free
    each one before the next is built: peak output memory is a single box-sized
    raster instead of ``n_time_points`` of them. The per-shape mechanism runs
    once up front (it also carries the anchor-exactness guard); only the
    per-pixel apply and the reshape to the full grid happen per time point.
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

    # The mechanism runs on a small per-shape table; the per-pixel apply
    # (covered_population * factor -> sum per pixel) is done one time point at
    # a time below so no covered_pixels x n_tps intermediate is ever materialized.
    shape_factors, shape_table = compute_shape_values(
        shape_id, coverage, population_total, prediction_data, pred_row,
        census_time_point, model_time_points, write_time_points,
        pooling_ids, rf_by_parent, bounds_by_parent, rf_country,
    )

    def raster_iter() -> Iterator[rt.RasterArray]:
        shape_id_series = pd.Series(shape_id)
        # Factorize pixel_id once (values are flattened raster positions). Per time
        # point we then sum-per-pixel with np.bincount and scatter into the box grid --
        # avoiding a groupby (re-hash) and a full-box reindex on every time point, which
        # is what made large admins slow.
        codes, covered_positions = pd.factorize(skel_pixel, sort=True)
        n_covered = len(covered_positions)
        size = template_raster.size
        for model_time_point in write_time_points:
            raking_factor = shape_id_series.map(
                shape_factors[f"raking_factor_{model_time_point}"]
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

    return shape_table, raster_iter()
