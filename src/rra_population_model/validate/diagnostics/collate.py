"""Collator: derived checks over the lane artifacts, the per-check summary, and the GATE exit.

Reads every lane artifact, computes the parquet-only checks (B2, B3, B4, B7, B8, B10,
C1, C3), evaluates the registry, and writes ``summary.md`` / ``summary.json`` under
``validation/diagnostics/``. Returns whether any GATE failed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.postprocess.raking_factors.runner import RAKING_VERSION
from rra_population_model.validate.diagnostics import artifacts, lane_c
from rra_population_model.validate.diagnostics.registry import (
    CHECKS,
    PIXEL_AREA_M2,
    Check,
)

PIXEL_AREA_M2_DEFAULT = PIXEL_AREA_M2["40"]
CENSUS_AGREEMENT_FACTOR = 3.0  # census within this factor of the model's own density counts as agreement

# Artifact ids behind each registry check; derived checks are computed here and saved as "release".
ARTIFACT_OF = {
    **{c: c for c in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "C2"]},
    "B1": "B1_scan", "B5": "B5", "B6": "B6", "B9": "B9",
}
DERIVED = ["B2", "B3", "B4", "B7", "B8", "B10", "C1", "C3"]
# Numeric columns worth summing into a check's headline when present.
HEADLINE_SUMS = [
    "people", "people_flagged", "people_vanished", "people_never_gated", "credit_removed",
    "post_anchor_credit", "people_single", "fringe_people", "wiped_people", "n_over_ppp_flag",
    "n_over_ppm3", "n_nan_introduced", "n_swing", "n_blink",
]


@dataclass
class CheckSummary:
    check_id: str
    severity: str
    unit_of_work: str
    description: str
    n_rows: int
    n_flagged: int
    n_gate: int
    gate_failed: bool
    headline: dict[str, float]
    artifact: str
    note: str = ""
    universe: int | None = None      # denominator for n_flagged (units, censuses, blocks ...), when known
    universe_label: str = ""
    guidance: str = ""


def summarize(
    check: Check, df: pd.DataFrame, artifact: str, note: str = "", universe: tuple[int, str] | None = None
) -> CheckSummary:
    if df.empty or "flag_class" not in df.columns:
        return CheckSummary(
            check_id=check.id, severity=check.severity, unit_of_work=check.unit_of_work, description=check.description,
            n_rows=0, n_flagged=0, n_gate=0, gate_failed=False, headline={}, artifact=artifact, note=note or "no rows",
            universe=universe[0] if universe else None, universe_label=universe[1] if universe else "",
            guidance=check.guidance,
        )
    flag_class = df["flag_class"]
    n_gate = int((flag_class == "GATE").sum())
    # artifacts partitioned by time point sum over 25 quarters; report per time point instead
    n_tps = df["time_point"].nunique() if "time_point" in df.columns else 0
    suffix = "/tp" if n_tps > 1 else ""
    headline = {
        f"{c}{suffix}": float(df[c].sum()) / (n_tps or 1)
        for c in HEADLINE_SUMS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    }
    if "abs_error" in df.columns:
        headline["max_abs_error"] = float(df["abs_error"].max())
    if "max" in df.columns and pd.api.types.is_numeric_dtype(df["max"]):
        headline["max_people_per_pixel"] = float(df["max"].max())
    return CheckSummary(
        check_id=check.id, severity=check.severity, unit_of_work=check.unit_of_work, description=check.description,
        n_rows=int(len(df)), n_flagged=int((flag_class != "REPORT").sum()), n_gate=n_gate,
        gate_failed=n_gate > 0 and check.severity == "GATE", headline=headline, artifact=artifact, note=note,
        universe=universe[0] if universe else None, universe_label=universe[1] if universe else "",
        guidance=check.guidance,
    )


def a8_relative_band(a8: pd.DataFrame) -> pd.DataFrame:
    """FLAG censuses whose rf_country sits outside [median / k, median * k] across all censuses."""
    if a8.empty:
        return a8
    k = CHECKS["A8"].thresholds["median_mult"]
    out = a8.copy()
    median = float(out["rf_country"].median())
    out["rf_country_median"] = median
    out["flag_class"] = np.where((out["rf_country"] > k * median) | (out["rf_country"] < median / k), "FLAG", "REPORT")
    return out


def b1_reclassify(scan: pd.DataFrame) -> pd.DataFrame:
    """B1 FLAG from exceedance *shares* rather than any single pixel; GATE rows are kept as scanned."""
    if scan.empty or "n_finite" not in scan.columns:
        return scan
    thr = CHECKS["B1"].thresholds
    out = scan.copy()
    exceed = out[["n_over_ppp_flag", "n_over_ppm3", "n_pop_no_volume"]].clip(lower=0).sum(axis=1)
    share = exceed / out["n_finite"].clip(lower=1)
    flag = (share > thr["exceedance_share"]) | (out["n_over_ppp_list"] > 0)
    out["flag_class"] = np.where(out["flag_class"] == "GATE", "GATE", np.where(flag, "FLAG", "REPORT"))
    return out


def b1_implausible(
    pixels: pd.DataFrame, attr: pd.DataFrame, names: pd.Series[str], crs: str
) -> pd.DataFrame:
    """Every scanned pixel above the plausibility line, with coordinates, residential volume per
    person and, where the block was attributed, the mechanism that put it there:
    - model density: the model itself predicts it (census absent or agreeing within 3x)
    - census amplifies model: the census unit holds far more people than the model predicted
    - census mass on low-volume pixel: the unit's people land on pixels with (almost) no stock
    Pixels with under m3_per_person_floor of volume per person are the physically impossible ones."""
    thr = CHECKS["B1"].thresholds
    columns = ["time_point", "block_key", "location_id", "location_name", "lon", "lat", "value", "raw", "volume_m3",
               "m3_per_person", "physically_impossible", "stage", "rf1", "rf2", "census_layer", "census_amplification",
               "census_iso3", "census_time_point", "mechanism", "flag_class"]
    hi = pixels[pixels["value"] > thr["ppp_list"]].drop(columns=["check_id"], errors="ignore").copy()
    if hi.empty:
        return pd.DataFrame(columns=columns)
    keys = ["time_point", "block_key", "x", "y"]
    if len(attr):
        hi = hi.merge(attr[[*keys, "rf1", "rf2", "census_layer", "stage", "census_iso3", "census_time_point"]], on=keys, how="left")
    else:
        for col in ["rf1", "rf2", "census_layer", "stage", "census_iso3", "census_time_point"]:
            hi[col] = np.nan
    points = gpd.GeoSeries(gpd.points_from_xy(hi["x"], hi["y"]), crs=crs).to_crs("EPSG:4326")
    hi["lon"], hi["lat"] = points.x.to_numpy(), points.y.to_numpy()
    hi["location_name"] = hi["location_id"].map(names)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi["m3_per_person"] = hi["volume_m3"] / hi["value"]
        hi["census_amplification"] = hi["value"] / (hi["raw"] * hi["rf1"] * hi["rf2"])
    hi["physically_impossible"] = hi["m3_per_person"] < thr["m3_per_person_floor"]
    low_volume = hi["volume_m3"] < PIXEL_AREA_M2_DEFAULT  # under 1 m3 of volume per m2 of pixel
    mechanism = np.select(
        [hi["stage"].isna(), hi["stage"] == "gbd", hi["census_amplification"] <= CENSUS_AGREEMENT_FACTOR, low_volume],
        ["unattributed", "model density (no census)", "model density, census agrees", "census mass on low-volume pixel"],
        default="census amplifies model",
    )
    hi["mechanism"] = mechanism
    hi["flag_class"] = np.where(hi["physically_impossible"], "FLAG", "REPORT")
    hi.insert(0, "check_id", "B1_implausible")
    return hi[["check_id", *columns]].sort_values("value", ascending=False).reset_index(drop=True)


def _b1_views(
    pm_data: PopulationModelData, resolution: str, scan: pd.DataFrame, pixels: pd.DataFrame, attr: pd.DataFrame, names: pd.Series[str]
) -> dict[str, pd.DataFrame]:
    """The two human-readable views of the pixel scan: per-location roll-up and the implausible-pixel list."""
    views = {"B1_locations": b1_by_location(scan, names)}
    if len(pixels):
        crs = pm_data.load_modeling_frame_info(resolution).crs
        views["B1_implausible"] = b1_implausible(pixels, attr, names, crs)
        per_location = views["B1_implausible"].groupby("location_id").agg(
            n_pixels_over_list=("value", "size"), n_physically_impossible=("physically_impossible", "sum")
        ).reset_index()
        views["B1_locations"] = views["B1_locations"].merge(per_location, on="location_id", how="left").fillna(
            {"n_pixels_over_list": 0, "n_physically_impossible": 0}
        )
    return views


def universes(loaded: dict[str, pd.DataFrame]) -> dict[str, tuple[int, str]]:
    """Denominators for the flagged counts, from the artifacts that enumerate each population."""
    a1, a8, scan, b5 = loaded.get("A1", pd.DataFrame()), loaded.get("A8", pd.DataFrame()), loaded.get("B1_scan", pd.DataFrame()), loaded.get("B5", pd.DataFrame())
    units = (int(a1["n_units"].sum()), "census units in tables") if len(a1) else None
    censuses = (int(len(a8)), "censuses") if len(a8) else None
    out: dict[str, tuple[int, str]] = {}
    for check_id in ["A2", "A3", "A4", "A5", "A9", "A11", "A12"]:
        if units:
            out[check_id] = units
    for check_id in ["A1", "A6", "A7"]:
        if len(a1):
            out[check_id] = (int(len(a1)), "census tasks")
    for check_id in ["A8", "A10", "C2"]:
        if censuses:
            out[check_id] = censuses
    if len(scan):
        out["B1"] = (int(len(scan)), "location-blocks x time points")
        out["B2"] = out["B7"] = (int(scan.groupby("time_point")["location_id"].nunique().sum()), "locations x time points")
    if len(b5):
        out["B5"] = (int(len(b5)), "blocks x time points")
    b6 = loaded.get("B6", pd.DataFrame())
    if len(b6):
        out["B6"] = (int(b6["block_key"].nunique()), "sampled blocks")
    return out


# ---------------------------------------------------------------- derived checks


def location_iso3(pm_data: PopulationModelData) -> pd.Series[str]:
    pop = pm_data.load_raking_population(version=RAKING_VERSION)
    pop = pop[pop["most_detailed"] == 1].drop_duplicates("location_id").set_index("location_id")
    return pop["ihme_loc_id"].str[:3]


def location_names(pm_data: PopulationModelData) -> pd.Series[str]:
    pop = pm_data.load_raking_population(version=RAKING_VERSION).drop_duplicates("location_id").set_index("location_id")
    return pop["location_name"]


def b1_by_location(scan: pd.DataFrame, names: pd.Series[str]) -> pd.DataFrame:
    """B1 rolled up per GBD location (mean per time point): the human-readable view of the pixel scan."""
    n_tps = max(scan["time_point"].nunique(), 1)
    sums = ["n_pixels", "n_finite", "n_over_ppp_flag", "n_over_ppp_list", "n_over_ppm3", "n_pop_no_volume", "n_nan_introduced", "n_inf", "n_negative"]
    grouped = scan.groupby("location_id")
    out = pd.DataFrame(grouped[sums].sum() / n_tps).rename(columns={c: f"{c}_per_tp" for c in sums})
    out["max_people_per_pixel"] = grouped["max"].max()
    out["n_gate"] = grouped["flag_class"].apply(lambda s: int((s == "GATE").sum()))
    out = out.reset_index()
    out.insert(1, "location_name", out["location_id"].map(names))
    out.insert(0, "check_id", "B1_locations")
    out["flag_class"] = "REPORT"
    return out.sort_values("n_over_ppp_flag_per_tp", ascending=False).reset_index(drop=True)


def _rf_columns(pm_data: PopulationModelData, model_spec: ModelSpecification, time_point: str, stage: str, columns: list[str]) -> pd.DataFrame:
    frame: pd.DataFrame = pd.read_parquet(pm_data.raking_factor_path(time_point, model_spec, stage=stage), columns=columns)
    return frame


def b2_reconciliation(
    pm_data: PopulationModelData, model_spec: ModelSpecification, scan: pd.DataFrame, blocks: pd.DataFrame, n_blocks: int
) -> pd.DataFrame:
    """Surface sums per location vs the GBD envelope. Only a complete scan of a time point can GATE."""
    tol = CHECKS["B2"].thresholds["rf_tol"]
    frames = []
    for tp, s in scan.groupby("time_point"):
        complete = blocks.loc[blocks["time_point"] == tp, "block_key"].nunique() >= n_blocks
        rf = _rf_columns(pm_data, model_spec, str(tp), "final", ["location_id", "true_pop", "raking_factor"])
        rf = rf.drop_duplicates("location_id").set_index("location_id")
        df = rf.join(s.groupby("location_id")["sum_final"].sum(), how="inner")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["deviation"] = df["sum_final"] / df["true_pop"] - 1
        checkable = np.isfinite(df["raking_factor"]) & (df["true_pop"] > 0)
        df["scan_complete"] = complete
        df["flag_class"] = np.where(complete & checkable & (df["deviation"].abs() > tol), "GATE", "REPORT")
        df["time_point"] = tp
        frames.append(df.reset_index())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out.insert(0, "check_id", "B2")
    return out


def b3_splice_smoothness(scan: pd.DataFrame, census_weights: pd.DataFrame, loc_iso3: pd.Series[str]) -> pd.DataFrame:
    """Adjacent-quarter change of location totals where the set of contributing censuses changes."""
    thr = CHECKS["B3"].thresholds
    totals = scan.groupby(["location_id", "time_point"])["sum_final"].sum().unstack("time_point")
    contributing = census_weights.reset_index().groupby(["iso3", "model_time_point"])["census_time_point"].apply(frozenset)
    rows: list[dict[str, Any]] = []
    for iso3 in contributing.index.get_level_values(0).unique():
        series = contributing.loc[iso3].sort_index()
        tps = list(series.index)
        locations = [loc for loc in totals.index if loc_iso3.get(loc) == iso3]
        for prev, cur in zip(tps[:-1], tps[1:], strict=False):
            if series[prev] == series[cur] or prev not in totals.columns or cur not in totals.columns:
                continue
            for loc in locations:
                before, after = float(totals.at[loc, prev]), float(totals.at[loc, cur])
                change = after - before
                rows.append({
                    "check_id": "B3", "iso3": iso3, "location_id": loc, "from_time_point": prev, "to_time_point": cur,
                    "censuses_from": ",".join(sorted(series[prev])), "censuses_to": ",".join(sorted(series[cur])),
                    "before": before, "after": after, "change": change,
                    "flag_class": "FLAG" if abs(change) > max(thr["rel"] * before, thr["abs"]) else "REPORT",
                })
    return pd.DataFrame(rows)


def b4_unmodeled(a12: pd.DataFrame) -> pd.DataFrame:
    if a12.empty:
        return pd.DataFrame()
    b = a12[a12["mismatch_class"] == "b_unmodeled"]
    grouped = b.groupby(["iso3", "census_time_point"]).agg(n_units=("shape_id", "size"), people=("census", "sum"))
    out = pd.DataFrame(grouped).reset_index()
    out.insert(0, "check_id", "B4")
    out["flag_class"] = "REPORT"
    return out


def b7_b8_splice(
    pm_data: PopulationModelData, model_spec: ModelSpecification, time_points: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """B7 fringe accounting and B8 census mass outside GBD masks from the persisted component sums.
    Both need the raking_factors prerequisite columns; without them the checks report as not computable."""
    if not time_points:
        return pd.DataFrame(), pd.DataFrame(), "no time points"
    names = pq.read_schema(pm_data.raking_factor_path(time_points[0], model_spec, stage="final")).names
    needed = {"block_raw", "block_census", "block_raw_census", "block_census_total"}
    if not needed <= set(names):
        return pd.DataFrame(), pd.DataFrame(), (
            "not computable: raking_factors/final lacks the component-sum columns; apply the raking_factors "
            "prerequisite patch and re-run `raking_factors --stage final`"
        )
    thr7, thr8 = CHECKS["B7"].thresholds, CHECKS["B8"].thresholds
    b7_frames, b8_frames = [], []
    for tp in time_points:
        rf = _rf_columns(pm_data, model_spec, tp, "final", ["block_key", "location_id", "true_pop", "raking_factor", *sorted(needed)])
        rf1 = _rf_columns(pm_data, model_spec, tp, "initial", ["location_id", "raking_factor"]).drop_duplicates("location_id").set_index("location_id")["raking_factor"]
        loc = rf.groupby("location_id").agg(true_pop=("true_pop", "first"), rf2=("raking_factor", "first"),
                                            raw=("block_raw", "sum"), census=("block_census", "sum"), raw_census=("block_raw_census", "sum"))
        loc["rf1"] = rf1.reindex(loc.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            loc["coverage_share"] = loc["raw_census"] / loc["raw"]
            loc["fringe_people"] = loc["rf1"] * (loc["raw"] - loc["raw_census"]) * loc["rf2"]
            loc["census_implied_factor"] = loc["census"] / loc["raw_census"]
            loc["ln_factor_ratio"] = np.log(loc["census_implied_factor"] / loc["rf1"])
            fringe_share = loc["fringe_people"] / loc["true_pop"]
        covered = loc["coverage_share"] >= thr7["min_coverage"]
        loc["flag_class"] = np.where(
            covered & (fringe_share > thr7["fringe_share"]) & (loc["ln_factor_ratio"].abs() > thr7["ln_factor_ratio"]), "FLAG", "REPORT"
        )
        loc["time_point"] = tp
        b7_frames.append(loc.reset_index())
        blocks = rf.groupby("block_key").agg(census_in_masks=("block_census", "sum"), census_total=("block_census_total", "first"))
        blocks["wiped_people"] = blocks["census_total"] - blocks["census_in_masks"]
        blocks["time_point"] = tp
        b8_frames.append(blocks.reset_index())
    b7 = pd.concat(b7_frames, ignore_index=True)
    b7.insert(0, "check_id", "B7")
    b8 = pd.concat(b8_frames, ignore_index=True)
    total_census = b8.groupby("time_point")["census_total"].transform("sum")
    b8["flag_class"] = np.where(b8["wiped_people"] > thr8["wiped_share"] * total_census, "FLAG", "REPORT")
    b8.insert(0, "check_id", "B8")
    return b7, b8, ""


def b10_rf_discontinuity(
    pm_data: PopulationModelData, model_spec: ModelSpecification, time_points: list[str],
    loc_iso3: pd.Series[str], census_iso3s: set[str]
) -> pd.DataFrame:
    """Final-factor ratio between adjacent GBD locations of the same census country."""
    thr = CHECKS["B10"].thresholds
    shapes = pm_data.load_raking_shapes(version=RAKING_VERSION)[["location_id", "geometry"]]
    shapes["iso3"] = shapes["location_id"].map(loc_iso3)
    shapes = shapes[shapes["iso3"].isin(census_iso3s)]
    pairs = gpd.sjoin(shapes, shapes[["location_id", "geometry"]], predicate="intersects", how="inner")
    pairs = pairs[pairs["location_id_left"] < pairs["location_id_right"]][["iso3", "location_id_left", "location_id_right"]]
    pairs = pairs[pairs["location_id_right"].map(loc_iso3) == pairs["iso3"]].drop_duplicates()
    if pairs.empty:
        return pd.DataFrame()
    frames = []
    for tp in time_points:
        rf = _rf_columns(pm_data, model_spec, tp, "final", ["location_id", "raking_factor"]).drop_duplicates("location_id").set_index("location_id")["raking_factor"]
        df = pairs.copy()
        df["rf_left"], df["rf_right"] = df["location_id_left"].map(rf), df["location_id_right"].map(rf)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ratio"] = np.maximum(df["rf_left"] / df["rf_right"], df["rf_right"] / df["rf_left"])
        df["time_point"] = tp
        df["above_ratio"] = df["ratio"] > thr["ratio"]
        frames.append(df)
    out: pd.DataFrame = pd.concat(frames, ignore_index=True)
    out.insert(0, "check_id", "B10")
    out["flag_class"] = "REPORT"
    return out


def c1_cross_release(
    pm_data: PopulationModelData, resolution: str, compare_version: str | None,
    scan: pd.DataFrame, parent_totals: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    if not compare_version:
        return pd.DataFrame(), "no --compare-version"
    prev_scan = artifacts.load_check(pm_data, resolution, compare_version, "B1_scan")
    prev_totals = artifacts.load_check(pm_data, resolution, compare_version, "A_parent_totals")
    frames = []
    if len(scan) and len(prev_scan):
        cur = scan.groupby(["time_point", "location_id"])["sum_final"].sum().reset_index()
        prev = prev_scan.groupby(["time_point", "location_id"])["sum_final"].sum().reset_index()
        frames.append(lane_c.cross_release(cur, prev, ["time_point", "location_id"], "sum_final").assign(level="gbd_location"))
    if len(parent_totals) and len(prev_totals):
        def anchor(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for (_, ctp), g in df.groupby(["iso3", "census_time_point"]):
                col = f"raked_{ctp}"
                if col in g.columns:
                    rows.append(g[["iso3", "census_time_point", "pooling_parent_id"]].assign(raked_anchor=g[col].to_numpy()))
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["iso3", "census_time_point", "pooling_parent_id", "raked_anchor"])
        frames.append(lane_c.cross_release(anchor(parent_totals), anchor(prev_totals),
                                           ["iso3", "census_time_point", "pooling_parent_id"], "raked_anchor").assign(level="census_parent"))
    if not frames:
        return pd.DataFrame(), f"{compare_version} has no diagnostics artifacts to compare against"
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "check_id", "C1")
    return out, ""


# ---------------------------------------------------------------- summary


def render_markdown(summaries: list[CheckSummary], version: str, compare_version: str | None, gate_failed: bool) -> str:
    lines = [
        f"# Diagnostics summary -- {version}",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Compare version: {compare_version or 'none'}. "
        f"**Release gate: {'FAILED' if gate_failed else 'passed'}.**",
        "",
        "| check | severity | status | rows | flagged (of universe) | headline | artifact |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        if s.gate_failed:
            status = f"**FAIL** ({s.n_gate} GATE rows)"
        elif s.n_rows == 0:
            status = f"n/a ({s.note})" if s.note else "n/a"
        elif s.n_flagged:
            status = f"FLAG {s.n_flagged:,}"
        else:
            status = "PASS"
        if s.universe:
            flagged = f"{s.n_flagged:,} of {s.universe:,} {s.universe_label} ({100 * s.n_flagged / s.universe:.2f}%)"
        else:
            flagged = f"{s.n_flagged:,}"
        headline = "; ".join(f"{k} {v:,.6g}" for k, v in s.headline.items()) or "-"
        lines.append(f"| {s.check_id} {s.description} | {s.severity} | {status} | {s.n_rows:,} | {flagged} | {headline} | `{s.artifact}` |")
    lines += ["", "## How to read each check", "",
              "Severity: GATE hard-fails the stage; FLAG is counted and listed, the release proceeds; REPORT is context. "
              "Flagged counts are shown against the population they were drawn from. Time-point artifacts report per quarter.", ""]
    for s in summaries:
        lines += [f"### {s.check_id} — {s.description}", "", s.guidance or "(no guidance recorded)", ""]
    return "\n".join(lines) + "\n"


def _derived_checks(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    resolution: str,
    version: str,
    lanes: set[str],
    compare_version: str | None,
    loaded: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The parquet-only checks computed from lane artifacts: B2, B3, B4, B7, B8, B10, C1, C3."""
    scan = loaded["B1_scan"]
    time_points = sorted(scan["time_point"].unique()) if len(scan) else []
    census_iso3s = set(pm_data.load_census_raking_tasks(model_spec)["iso3"].unique())
    derived: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}
    if "b" in lanes and len(scan):
        n_blocks = pm_data.load_modeling_frame(resolution)["block_key"].nunique()
        loc_iso3 = location_iso3(pm_data)
        census_weights = pd.read_parquet(pm_data.raked_census_root(resolution, version) / "weights.parquet")
        derived["B2"] = b2_reconciliation(pm_data, model_spec, scan, loaded["B5"], n_blocks)
        derived["B3"] = b3_splice_smoothness(scan, census_weights, loc_iso3)
        derived["B7"], derived["B8"], note = b7_b8_splice(pm_data, model_spec, time_points)
        if note:
            notes["B7"] = notes["B8"] = note
        derived["B10"] = b10_rf_discontinuity(pm_data, model_spec, time_points, loc_iso3, census_iso3s)
    derived["B4"] = b4_unmodeled(loaded["A12"])
    if "c" in lanes:
        derived["C1"], c1_note = c1_cross_release(pm_data, resolution, compare_version, scan, loaded["A_parent_totals"])
        if c1_note:
            notes["C1"] = c1_note
        derived["C3"] = lane_c.held_out_scores(loaded["A_parent_totals"]) if len(loaded["A_parent_totals"]) else pd.DataFrame()
    return derived, notes


def collate_main(
    resolution: str, version: str, compare_version: str | None, output_dir: str, lanes: set[str] | None = None
) -> bool:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    lanes = lanes or {"a", "b", "c"}

    def load(check_id: str) -> pd.DataFrame:
        return artifacts.load_check(pm_data, resolution, version, check_id)

    loaded = {aid: load(aid) for aid in {*ARTIFACT_OF.values(), "A12", "A_parent_totals", "B5"}}
    loaded["A8"] = a8_relative_band(loaded["A8"])
    loaded["B1_scan"] = b1_reclassify(loaded["B1_scan"])
    attr = load("B1_attr")
    if len(attr) and len(loaded["B1_scan"]):
        # splice reconstruction mismatches from B-attribute count against B1
        mismatches = attr[attr["flag_class"] == "GATE"].assign(check_id="B1")
        loaded["B1_scan"] = pd.concat([loaded["B1_scan"], mismatches], ignore_index=True)

    derived, notes = _derived_checks(pm_data, model_spec, resolution, version, lanes, compare_version, loaded)
    names = location_names(pm_data)
    for frame in derived.values():
        if len(frame) and "location_id" in frame.columns and "location_name" not in frame.columns:
            frame["location_name"] = frame["location_id"].map(names)
    if len(loaded["B1_scan"]):
        derived.update(_b1_views(pm_data, resolution, loaded["B1_scan"], load("B1_pixels"), attr, names))
    for check_id, frame in derived.items():
        if len(frame):
            artifacts.save_check(pm_data, frame, resolution, version, check_id, "release")

    denominators = universes(loaded)
    summaries = []
    for check in CHECKS.values():
        if not check.enabled:
            continue
        artifact_id = ARTIFACT_OF.get(check.id, check.id)
        frame = derived.get(check.id, loaded.get(artifact_id, pd.DataFrame()))
        location = str(artifacts.check_dir(pm_data, resolution, version, artifact_id))
        summaries.append(summarize(check, frame, location, notes.get(check.id, ""), denominators.get(check.id)))
    gate_failed = any(s.gate_failed for s in summaries)

    artifacts.diagnostics_root(pm_data, resolution, version).mkdir(parents=True, exist_ok=True)
    artifacts.summary_path(pm_data, resolution, version, "json").write_text(json.dumps({
        "version": version, "compare_version": compare_version,
        "generated": datetime.now(UTC).isoformat(), "gate_failed": gate_failed,
        "checks": [asdict(s) for s in summaries],
    }, indent=2))
    markdown = render_markdown(summaries, version, compare_version, gate_failed)
    artifacts.summary_path(pm_data, resolution, version, "md").write_text(markdown)
    print(markdown)
    return gate_failed
