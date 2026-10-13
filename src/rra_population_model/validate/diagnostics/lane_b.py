"""Lane B: final-surface checks over raked_predictions.

B-scan: one task per time point streams every block (B1 pixel plausibility, B5 bounds
sanity, per-location surface sums for B2). B-attribute: one task per flagged or sampled
block rebuilds the splice components at the flagged pixels through the production
helpers (B1 attribution) and runs the per-pixel trajectory (B6) and multi-census
partial-coverage (B9) passes. Nothing here re-derives the mechanism.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
import shapely
from rasterra._features import raster_geometry_mask
from rra_tools import parallel

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.postprocess.rake.runner import build_raking_factor_raster
from rra_population_model.postprocess.raking_factors.runner import RAKING_VERSION
from rra_population_model.postprocess.utils import (
    block_census_tasks,
    get_prediction_time_point,
    load_block_census_layer,
    paste_on_canvas,
)
from rra_population_model.validate.diagnostics import artifacts
from rra_population_model.validate.diagnostics.registry import (
    CHECKS,
    PIXEL_AREA_M2,
    TOP_K_PER_BLOCK,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Bounds = tuple[float, float, float, float]
GRID_SNAP_TOLERANCE = 1e-6  # pixels, for B5 grid alignment
RECONSTRUCTION_TOLERANCE = 1e-5  # relative; float32 product vs float64 rebuild
MIN_QUARTERS_FOR_SWING = 2
MIN_CENSUSES_FOR_COVERAGE = 2


class ScanArgs(NamedTuple):
    """Identity only (no geometries) so the pickled payload per worker stays small."""

    resolution: str
    version: str
    block_key: str
    time_point: str
    output_dir: str
    block_bounds: Bounds


def _align(raster: rt.RasterArray, like: rt.RasterArray) -> FloatArray:
    """``raster`` on ``like``'s exact grid as float64 (a no-op when the grids already agree)."""
    if raster.shape == like.shape and raster.transform == like.transform:
        return raster.to_numpy().astype(np.float64)
    return paste_on_canvas(raster, like).to_numpy().astype(np.float64)


def _location_masks(
    rf_rows: gpd.GeoDataFrame, like: rt.RasterArray
) -> list[tuple[int, float, BoolArray]]:
    masks: list[tuple[int, float, BoolArray]] = []
    for location_id, rf, geom in rf_rows[["location_id", "raking_factor", "geometry"]].itertuples(index=False):
        mask, *_ = raster_geometry_mask(
            data_transform=like.transform,
            data_width=like.shape[1],
            data_height=like.shape[0],
            shapes=[geom],
            invert=True,
        )
        masks.append((int(location_id), float(rf), np.asarray(mask, dtype=bool)))
    return masks


def _pixel_xy(like: rt.RasterArray, rows: npt.NDArray[np.intp], cols: npt.NDArray[np.intp]) -> tuple[FloatArray, FloatArray]:
    xs, ys = like.transform * (cols + 0.5, rows + 0.5)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _quantile(values: FloatArray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def scan_block(args: ScanArgs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B1 scan of one block at one time point.

    Returns (per-location rows, the block's top-K pixels, one block-level row for B5).
    Per-location rows carry the GATE/FLAG pixel counts, the people-per-pixel tail and the
    surface sum the collator reconciles against the GBD envelope (B2)."""
    pm_data = PopulationModelData(args.output_dir)
    model_spec = pm_data.load_model_specification(args.resolution, args.version)
    pred_tp = get_prediction_time_point(pm_data, args.resolution, args.version, args.time_point)
    final = pm_data.load_raked_prediction(args.block_key, args.time_point, model_spec)
    arr = final.to_numpy().astype(np.float64)
    finite = np.isfinite(arr)
    raw = _align(pm_data.load_raw_prediction(args.block_key, pred_tp, model_spec), final)
    try:
        volume = _align(
            pm_data.load_feature(args.resolution, args.block_key, model_spec.denominator, pred_tp), final
        )
        volume = np.where(np.isfinite(volume), volume, 0.0)
    except FileNotFoundError:
        volume = None
    rf_rows = pm_data.load_raking_factors(
        args.time_point, model_spec, stage="final", filters=[("block_key", "==", args.block_key)]
    )
    thr = CHECKS["B1"].thresholds
    scale = PIXEL_AREA_M2[args.resolution] / PIXEL_AREA_M2["40"]
    if volume is not None:
        # the denominator feature is residential volume per m2 of pixel (m3/m2); convert to m3
        volume = volume * PIXEL_AREA_M2[args.resolution]
        with np.errstate(divide="ignore", invalid="ignore"):
            ppm3 = np.where(volume > 0, arr / volume, np.nan)

    location_raster = np.full(arr.shape, -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for location_id, rf, mask in _location_masks(rf_rows, final):
        location_raster[mask] = location_id
        f = mask & finite
        v = arr[f]
        row: dict[str, Any] = {
            "time_point": args.time_point, "block_key": args.block_key, "location_id": location_id,
            "rf_final": rf, "rf_final_finite": bool(np.isfinite(rf)),
            "n_pixels": int(mask.sum()), "n_finite": int(f.sum()),
            "sum_final": float(v.sum()),
            "n_inf": int(np.isinf(arr[mask]).sum()), "n_negative": int((v < 0).sum()),
            # NaN in the product where the raw prediction was finite: legitimate only where the
            # location's final factor is itself non-finite (unmodeled carve-out, B4/B8)
            "n_nan_introduced": int((mask & np.isnan(arr) & np.isfinite(raw)).sum()),
            "n_over_ppp_flag": int((v > thr["ppp_flag"] * scale).sum()),
            "n_over_ppp_list": int((v > thr["ppp_list"] * scale).sum()),
            "q999": _quantile(v, 0.999), "q9999": _quantile(v, 0.9999),
            "max": float(v.max()) if v.size else float("nan"),
        }
        if volume is not None:
            row["n_over_ppm3"] = int((ppm3[f] > thr["ppm3"]).sum())
            row["n_pop_no_volume"] = int((f & (arr > 0) & (volume == 0)).sum())
        else:
            row["n_over_ppm3"] = row["n_pop_no_volume"] = -1  # denominator raster unavailable
        rows.append(row)
    scan = pd.DataFrame(rows)
    if len(scan):
        gate = (scan["n_inf"] + scan["n_negative"] > 0) | ((scan["n_nan_introduced"] > 0) & scan["rf_final_finite"])
        flag = scan[["n_over_ppp_flag", "n_over_ppm3", "n_pop_no_volume"]].clip(lower=0).sum(axis=1) > 0
        scan["flag_class"] = np.where(gate, "GATE", np.where(flag, "FLAG", "REPORT"))
    scan.insert(0, "check_id", "B1")

    k = min(TOP_K_PER_BLOCK, int(finite.sum()))
    flat = np.where(finite, arr, -np.inf).ravel()
    top = np.argpartition(flat, -k)[-k:] if k else np.array([], dtype=np.intp)
    r, c = np.unravel_index(top, arr.shape)
    xs, ys = _pixel_xy(final, r, c)
    pixels = pd.DataFrame({
        "check_id": "B1_pixels", "time_point": args.time_point, "block_key": args.block_key,
        "row": r, "col": c, "x": xs, "y": ys, "value": arr[r, c],
        "raw": raw[r, c], "volume_m3": volume[r, c] if volume is not None else np.nan,
        "location_id": location_raster[r, c],
    })

    block_row = _bounds_row(final, args, n_finite=int(finite.sum()), n_locations=len(rows))
    return scan, pixels, block_row


def _bounds_row(final: rt.RasterArray, args: ScanArgs, n_finite: int, n_locations: int) -> pd.DataFrame:
    """B5: raster bounds ordered, inside the block's frame bounds, and snapped to the block grid."""
    x_min, x_max, y_min, y_max = final.bounds  # rasterra orders bounds (x_min, x_max, y_min, y_max)
    bx_min, by_min, bx_max, by_max = args.block_bounds  # shapely total_bounds order
    res_x, _ = final.resolution
    res = float(abs(res_x))
    offset = (x_min - bx_min) / res
    snapped = abs(offset - round(offset)) < GRID_SNAP_TOLERANCE
    within = (x_min >= bx_min - res) and (x_max <= bx_max + res) and (y_min >= by_min - res) and (y_max <= by_max + res)
    ordered = (x_min < x_max) and (y_min < y_max) and (res_x > 0)
    return pd.DataFrame([{
        "check_id": "B5", "time_point": args.time_point, "block_key": args.block_key,
        "n_pixels": int(final.shape[0] * final.shape[1]), "n_finite": n_finite, "n_locations": n_locations,
        "bounds_ordered": bool(ordered), "within_block": bool(within), "grid_snapped": bool(snapped),
        "flag_class": "REPORT" if (ordered and within and snapped) else "FLAG",
    }])


def block_bounds(model_frame: gpd.GeoDataFrame) -> dict[str, Bounds]:
    frame_bounds = model_frame.groupby("block_key")["geometry"].apply(lambda g: g.total_bounds)
    return {bk: tuple(float(v) for v in b) for bk, b in frame_bounds.items()}  # type: ignore[misc]


def scan_main(
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
    block_keys: list[str] | None = None,
) -> None:
    """Stream every block of one time point through scan_block and write B1_scan, B1_pixels, B5."""
    pm_data = PopulationModelData(output_dir)
    bounds = block_bounds(pm_data.load_modeling_frame(resolution))
    keys = block_keys if block_keys is not None else sorted(bounds)
    args = [ScanArgs(resolution, version, bk, time_point, output_dir, bounds[bk]) for bk in keys]
    print(f"scanning {len(args)} blocks at {time_point}")
    results = parallel.run_parallel(scan_block, args, num_cores=num_cores, progress_bar=progress_bar)
    scan = pd.concat([r[0] for r in results if len(r[0])], ignore_index=True)
    pixels = pd.concat([r[1] for r in results if len(r[1])], ignore_index=True)
    blocks = pd.concat([r[2] for r in results], ignore_index=True)
    artifacts.save_check(pm_data, scan, resolution, version, "B1_scan", time_point)
    artifacts.save_check(pm_data, pixels, resolution, version, "B1_pixels", time_point)
    artifacts.save_check(pm_data, blocks, resolution, version, "B5", time_point)
    print(
        f"{time_point}: {len(scan):,} location-blocks; GATE {int((scan['flag_class'] == 'GATE').sum()):,}; "
        f"FLAG {int((scan['flag_class'] == 'FLAG').sum()):,}; B5 flags {int((blocks['flag_class'] == 'FLAG').sum()):,}"
    )


# ---------------------------------------------------------------- B-attribute


def _census_components(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    block_geometry: shapely.Polygon | shapely.MultiPolygon,
    prediction_time_point: str,
    census_tasks: list[tuple[str, str, str]],
    census_weights: pd.DataFrame,
) -> list[tuple[tuple[str, str, str], rt.RasterArray]]:
    """The weighted per-task rasters behind the census layer.

    Calls the production splice helper once per task (a one-task list returns that
    task's weighted, block-clipped raster, or None when it contributes nothing), so the
    components are exactly what the rake stage merges -- no parallel implementation.
    """
    components = []
    for task in census_tasks:
        layer = load_block_census_layer(
            pm_data, model_spec, block_geometry, prediction_time_point, [task], census_weights
        )
        if layer is not None:
            components.append((task, layer))
    return components


def _finite_at(raster: rt.RasterArray, x: float, y: float) -> bool:
    """Whether ``raster`` holds a finite value at map coordinate (x, y), indexed on its own grid."""
    col, row = ~raster.transform * (x, y)
    r, c = int(np.floor(row)), int(np.floor(col))
    if not (0 <= r < raster.shape[0] and 0 <= c < raster.shape[1]):
        return False
    return bool(np.isfinite(raster.to_numpy()[r, c]))


def _rowcol(like: rt.RasterArray, xs: FloatArray, ys: FloatArray) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    cols, rows = ~like.transform * (xs, ys)
    return np.floor(np.asarray(rows)).astype(np.intp), np.floor(np.asarray(cols)).astype(np.intp)


def attribute_block(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    block_key: str,
    block_geometry: shapely.Polygon | shapely.MultiPolygon,
    pixels: pd.DataFrame,
    census_tasks: list[tuple[str, str, str]],
    census_weights: pd.DataFrame,
) -> pd.DataFrame:
    """For each flagged pixel: raw, rf1, census-layer value (and the census task that supplied it),
    rf2, and the reconstruction rf2 * (census if finite else rf1 * raw), which must match the
    persisted final at float32 precision -- a free consistency check on the splice itself."""
    resolution, version = model_spec.resolution, model_spec.model_version
    out = []
    for tp, group in pixels.groupby("time_point"):
        pred_tp = get_prediction_time_point(pm_data, resolution, version, str(tp))
        raw = pm_data.load_raw_prediction(block_key, pred_tp, model_spec)
        rf1_rows = pm_data.load_raking_factors(str(tp), model_spec, stage="initial", filters=[("block_key", "==", block_key)])
        rf2_rows = pm_data.load_raking_factors(str(tp), model_spec, stage="final", filters=[("block_key", "==", block_key)])
        rf1 = build_raking_factor_raster(rf1_rows, raw).to_numpy() if len(rf1_rows) else np.full(raw.shape, np.nan)
        rf2 = build_raking_factor_raster(rf2_rows, raw).to_numpy() if len(rf2_rows) else np.full(raw.shape, np.nan)
        components = _census_components(pm_data, model_spec, block_geometry, pred_tp, census_tasks, census_weights)
        layer = (
            paste_on_canvas(rt.merge([raster for _, raster in components], method="sum"), raw).to_numpy()
            if components else np.full(raw.shape, np.nan)
        )
        raw_arr = raw.to_numpy().astype(np.float64)
        rows_idx, cols_idx = _rowcol(raw, group["x"].to_numpy(dtype=float), group["y"].to_numpy(dtype=float))
        for (r, c), (_, px) in zip(zip(rows_idx, cols_idx, strict=True), group.iterrows(), strict=True):
            inside = 0 <= r < raw.shape[0] and 0 <= c < raw.shape[1]
            census_value = float(layer[r, c]) if inside else np.nan
            # index each component on its own (block-clipped) grid: pasting every component onto the
            # full block canvas cost ~270 MB per census task and blew past 36G on dense blocks
            supplier = next((task for task, raster in components if inside and _finite_at(raster, px["x"], px["y"])), None)
            rf1_v, rf2_v, raw_v = (float(rf1[r, c]), float(rf2[r, c]), float(raw_arr[r, c])) if inside else (np.nan,) * 3
            base = census_value if np.isfinite(census_value) else rf1_v * raw_v
            out.append({
                "check_id": "B1_attr", "time_point": tp, "block_key": block_key,
                "x": px["x"], "y": px["y"], "location_id": px["location_id"], "final": px["value"],
                "raw": raw_v, "rf1": rf1_v, "census_layer": census_value, "rf2": rf2_v,
                "census_iso3": supplier[0] if supplier else None,
                "census_task_parent_id": supplier[1] if supplier else None,
                "census_time_point": supplier[2] if supplier else None,
                "stage": "census" if np.isfinite(census_value) else ("gbd" if np.isfinite(rf1_v * raw_v) else "nodata"),
                "reconstructed": rf2_v * base,
            })
    result = pd.DataFrame(out)
    if len(result):
        result["reconstruction_error"] = (result["reconstructed"] - result["final"]).abs() / result["final"].abs().clip(lower=1e-12)
        # float32 product vs float64 reconstruction: agreement beyond 1e-5 is a splice discrepancy
        result["flag_class"] = np.where(result["reconstruction_error"] > RECONSTRUCTION_TOLERANCE, "GATE", "REPORT")
    return result


def _trajectory_stats(stack: FloatArray, swing_thr: float, blink_thr: float) -> dict[str, Any]:
    """Per-pixel swing, blinks and steps for one (time, rows, cols) stack of final values."""
    positive = np.nan_to_num(stack, nan=0.0) > 0
    ever = positive.any(axis=0)
    n_quarters = stack.shape[0]
    idx = np.arange(n_quarters)[:, None, None]
    first_on = positive.argmax(axis=0)
    last_on = n_quarters - 1 - positive[::-1].argmax(axis=0)
    between = (idx > first_on[None]) & (idx < last_on[None])
    blinks = np.where(ever, ((~positive) & between).sum(axis=0), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_max = np.where(positive, stack, -np.inf).max(axis=0)
        pos_min = np.where(positive, stack, np.inf).min(axis=0)
        swing = np.where(positive.sum(axis=0) >= MIN_QUARTERS_FOR_SWING, pos_max / pos_min, 1.0)
        if n_quarters > 1:
            a, b = stack[:-1], stack[1:]
            both = positive[:-1] & positive[1:]
            step = np.where(both, np.maximum(a / b, b / a), 1.0).max(axis=0)
        else:
            step = np.ones(ever.shape)
    swing_flag = ever & (swing > swing_thr)
    return {
        "n_pixels_ever_positive": int(ever.sum()),
        "n_swing": int(swing_flag.sum()),
        "n_blink": int((blinks > blink_thr).sum()),
        "n_step": int((ever & (step > swing_thr)).sum()),
        "people_swing_last": float(np.nan_to_num(stack[-1])[swing_flag].sum()),
        "swings": swing[ever],
        "blinks_max": int(blinks.max()) if ever.any() else 0,
    }


def trajectory_block(
    pm_data: PopulationModelData, model_spec: ModelSpecification, block_key: str, time_points: list[str], chunk_rows: int = 512
) -> pd.DataFrame:
    """B6: per-pixel swing, blink count and largest one-quarter step over the window, summarized per
    block. Rasters are read in row chunks so a full 25-quarter block never sits in memory at once."""
    thr = CHECKS["B6"].thresholds
    rasters = [pm_data.load_raked_prediction(block_key, tp, model_spec) for tp in time_points]
    arrays = [r.to_numpy() for r in rasters]
    n_rows = arrays[0].shape[0]
    totals: dict[str, float] = {"n_pixels_ever_positive": 0, "n_swing": 0, "n_blink": 0, "n_step": 0, "people_swing_last": 0.0}
    swings, blinks_max = [], 0
    for start in range(0, n_rows, chunk_rows):
        stack = np.stack([a[start : start + chunk_rows].astype(np.float64) for a in arrays])
        stats = _trajectory_stats(stack, thr["swing"], thr["max_blinks"])
        for key in totals:
            totals[key] += stats[key]
        swings.append(stats["swings"])
        blinks_max = max(blinks_max, stats["blinks_max"])
    swing_values = np.concatenate(swings) if swings else np.array([])
    row = {
        "check_id": "B6", "block_key": block_key, "n_time_points": len(time_points), **totals,
        "swing_q99": _quantile(swing_values, 0.99),
        "swing_max": float(swing_values.max()) if swing_values.size else float("nan"),
        "blinks_max": blinks_max,
        "flag_class": "FLAG" if (totals["n_swing"] or totals["n_blink"]) else "REPORT",
    }
    return pd.DataFrame([row])


def partial_coverage_block(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    block_key: str,
    block_geometry: shapely.Polygon | shapely.MultiPolygon,
    time_points: list[str],
    census_tasks: list[tuple[str, str, str]],
    census_weights: pd.DataFrame,
) -> pd.DataFrame:
    """B9: where >= 2 censuses of one country contribute to a time point, count the pixels valid in
    exactly one of them (the merge-sum carries that census's weight alone) and the people on them."""
    resolution, version = model_spec.resolution, model_spec.model_version
    weights = census_weights.reset_index()
    rows = []
    for tp in time_points:
        pred_tp = get_prediction_time_point(pm_data, resolution, version, tp)
        contributing = weights[weights["model_time_point"] == pred_tp].groupby("iso3")["census_time_point"].apply(set)
        multi = {str(iso3) for iso3, cs in contributing.items() if len(cs) >= MIN_CENSUSES_FOR_COVERAGE and any(t[0] == iso3 for t in census_tasks)}
        if not multi:
            continue
        final = pm_data.load_raked_prediction(block_key, tp, model_spec)
        arr = final.to_numpy().astype(np.float64)
        components = _census_components(pm_data, model_spec, block_geometry, pred_tp, census_tasks, census_weights)
        for iso3 in sorted(multi):
            valid_by_census: dict[str, npt.NDArray[np.bool_]] = {}
            for (c_iso3, _, c_tp), raster in components:
                if c_iso3 != iso3:
                    continue
                finite = np.isfinite(paste_on_canvas(raster, final).to_numpy())
                valid_by_census[c_tp] = valid_by_census.get(c_tp, np.zeros(arr.shape, dtype=bool)) | finite
            if len(valid_by_census) < MIN_CENSUSES_FOR_COVERAGE:
                continue
            n_valid = np.sum(list(valid_by_census.values()), axis=0)
            single = n_valid == 1
            rows.append({
                "check_id": "B9", "time_point": tp, "block_key": block_key, "iso3": iso3,
                "censuses": ",".join(sorted(str(k) for k in valid_by_census)),
                "n_pixels_any": int((n_valid > 0).sum()), "n_pixels_single": int(single.sum()),
                "people_single": float(np.nan_to_num(arr)[single].sum()),
                "people_any": float(np.nan_to_num(arr)[n_valid > 0].sum()),
            })
    out = pd.DataFrame(rows)
    if len(out):
        share = out["people_single"] / out["people_any"].clip(lower=1e-12)
        out["flag_class"] = np.where(share > CHECKS["B9"].thresholds["people_share"], "FLAG", "REPORT")
    return out


def census_inputs_for_block(
    pm_data: PopulationModelData, model_spec: ModelSpecification, block_key: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Task admins and weights for the countries whose GBD locations the scan saw in this block;
    the global task_admins parquet (~1.6 GB) is only loaded when the scan has no rows for it."""
    scan = artifacts.load_check(pm_data, model_spec.resolution, model_spec.model_version, "B1_scan")
    locations = set(scan.loc[scan["block_key"] == block_key, "location_id"]) if len(scan) else set()
    if not locations:
        return pm_data.load_census_raking_inputs(model_spec)
    population = pm_data.load_raking_population(version=RAKING_VERSION).drop_duplicates("location_id").set_index("location_id")
    iso3s = sorted({str(population["ihme_loc_id"].get(loc, ""))[:3] for loc in locations} - {""})
    censused = set(pm_data.load_census_raking_tasks(model_spec)["iso3"])
    parts = [pm_data.load_census_raking_inputs(model_spec, iso3=iso3) for iso3 in iso3s if iso3 in censused]
    if not parts:
        empty_admins, empty_weights = pm_data.load_census_raking_inputs(model_spec, iso3="___")
        return empty_admins, empty_weights
    admins = pd.concat([p[0] for p in parts])
    weights = pd.concat([p[1] for p in parts])
    return gpd.GeoDataFrame(admins, geometry="geometry", crs=parts[0][0].crs), weights


def attribute_main(
    block_key: str, time_point: str, trajectory_sample: bool, resolution: str, version: str, output_dir: str
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    model_frame = pm_data.load_modeling_frame(resolution)
    block_geometry = model_frame.loc[model_frame["block_key"] == block_key].union_all()
    time_points = (
        pm_data.list_raked_prediction_time_points(resolution, version) if time_point == "ALL" else [time_point]
    )
    pixels = artifacts.load_check(pm_data, resolution, version, "B1_pixels")
    pixels = pixels[(pixels["block_key"] == block_key) & pixels["time_point"].isin(time_points)] if len(pixels) else pixels
    task_admins, census_weights = census_inputs_for_block(pm_data, model_spec, block_key)
    census_tasks = block_census_tasks(task_admins, block_geometry)

    attribution = attribute_block(pm_data, model_spec, block_key, block_geometry, pixels, census_tasks, census_weights)
    if len(attribution) and (attribution["flag_class"] == "GATE").any():
        # A reconstruction mismatch usually means the country lookup missed a census (Hong Kong is
        # CHN_354 in GBD while its census is HKG): retry with every task admin touching the block.
        print(f"{block_key}: reconstruction mismatches with country-filtered census inputs; retrying with the global task admins")
        task_admins, census_weights = pm_data.load_census_raking_inputs(model_spec)
        census_tasks = block_census_tasks(task_admins, block_geometry)
        attribution = attribute_block(pm_data, model_spec, block_key, block_geometry, pixels, census_tasks, census_weights)
    artifacts.save_check(pm_data, attribution, resolution, version, "B1_attr", block_key)
    print(f"{block_key}: attributed {len(attribution)} pixels; splice mismatches {int((attribution['flag_class'] == 'GATE').sum()) if len(attribution) else 0}")
    if trajectory_sample:
        artifacts.save_check(pm_data, trajectory_block(pm_data, model_spec, block_key, time_points), resolution, version, "B6", block_key)
        coverage = partial_coverage_block(pm_data, model_spec, block_key, block_geometry, time_points, census_tasks, census_weights)
        artifacts.save_check(pm_data, coverage, resolution, version, "B9", block_key)
        print(f"{block_key}: B6 written; B9 rows {len(coverage)}")
