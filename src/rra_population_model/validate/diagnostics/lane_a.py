"""Lane A: per-census checks over the persisted census-rake artifacts.

One task per (iso3, census_time_point). Reads the per-shape tables, the rf_prior,
the census file (no geometry) and, for the unmodeled-location class, the final
raking factors; never recomputes the mechanism. Writes one artifact per check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.postprocess.raking_factors.runner import RAKING_VERSION
from rra_population_model.validate.diagnostics import artifacts, lane_c
from rra_population_model.validate.diagnostics.registry import (
    CHECKS,
    MIN_FAMILY_UNITS,
)

MIN_QUARTERS_FOR_SWING = 2  # a swing needs two positive quarters
UNIT_COLUMNS = ["shape_id", "parent_id", "admin_level", "shape_name", "path_to_top_parent", "population_total"]
FloatArray = npt.NDArray[np.float64]


class CensusUnit(NamedTuple):
    iso3: str
    census_time_point: str


def census_year(census_time_point: str) -> str:
    return census_time_point.split("q")[0]


def written_time_points(table: pd.DataFrame) -> list[str]:
    """Sorted time points from the raked_{tp} columns; the census tp is always among them."""
    return sorted(c[len("raked_") :] for c in table.columns if c.startswith("raked_"))


def with_keys(
    df: pd.DataFrame, check_id: str, unit: CensusUnit, task_parent_id: str | None = None
) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "check_id", check_id)
    df.insert(1, "iso3", unit.iso3)
    df.insert(2, "census_time_point", unit.census_time_point)
    if task_parent_id is not None:
        df.insert(3, "task_parent_id", task_parent_id)
    return df


def _col(table: pd.DataFrame, name: str) -> FloatArray:
    return table[name].to_numpy(dtype=np.float64)


def _stack(table: pd.DataFrame, prefix: str, time_points: list[str]) -> FloatArray:
    return np.column_stack([_col(table, f"{prefix}_{tp}") for tp in time_points])


def uncapped_credit(table: pd.DataFrame, time_point: str, census_time_point: str) -> FloatArray:
    """Credit the cap removed at ``time_point``: y_unc - raked, with
    y_unc = 1{shape_population > 0} * (C + rf_parent * built * built / (built + base)) for t >= t_c
    (exact reconstruction of the mechanism's uncapped y; zero before the anchor)."""
    if time_point < census_time_point:
        return np.zeros(len(table))
    census, base = _col(table, "census_population"), _col(table, "base_population")
    built, rf = _col(table, f"built_{time_point}"), _col(table, "rf_parent")
    predicted = _col(table, f"shape_population_{time_point}")
    weight = np.where(built + base > 0, built / np.maximum(built + base, 1e-300), 0.0)
    y_unc = np.where(predicted > 0, census + rf * built * weight, 0.0)
    return y_unc - _col(table, f"raked_{time_point}")


# ---------------------------------------------------------------- per-task checks


def check_a1_anchor(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> pd.DataFrame:
    """Sum of gated raked(t_c) vs sum of gated census; gated = base > 0, the in-task guard's definition."""
    thr = CHECKS["A1"].thresholds
    ctp = unit.census_time_point
    gated = _col(table, "base_population") > 0
    census = float(_col(table, "census_population")[gated].sum())
    raked = float(_col(table, f"raked_{ctp}")[gated].sum())
    err = abs(raked - census)
    tol = max(thr["rel_tol"] * census, thr["abs_floor"])
    row = {
        "n_units": len(table), "n_gated": int(gated.sum()),
        "census_gated": census, "raked_gated": raked, "abs_error": err,
        # not-<= so a NaN error fails loudly, like the guard
        "flag_class": "GATE" if not (err <= tol) else "REPORT",
    }
    return with_keys(pd.DataFrame([row]), "A1", unit, task_parent_id)


def check_a2_growth_density(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> pd.DataFrame:
    """Implied density raked/n_on against the unit's density_bound.

    A *credited* unit (raked > C) above its bound is a cap regression -> GATE. A unit whose
    established density C/n_on(t_c) already exceeds the bound is the accepted at-anchor
    extreme (the cap escapes via max(C, .)) -> REPORT, listed once at the anchor."""
    ctp = unit.census_time_point
    census, bound = _col(table, "census_population"), _col(table, "density_bound")
    finite = np.isfinite(bound)
    frames = []
    for tp in written_time_points(table):
        if tp < ctp:
            continue
        raked, n_on = _col(table, f"raked_{tp}"), _col(table, f"n_on_{tp}")
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.where(n_on > 0, raked / n_on, np.nan)
        credited = raked > census * (1 + 1e-9) + 1e-9
        exceed = finite & (implied > bound * (1 + 1e-6))
        sel = exceed & credited
        if tp == ctp:
            with np.errstate(divide="ignore", invalid="ignore"):
                anchor_density = np.where(n_on > 0, census / n_on, np.nan)
            sel_anchor = finite & (anchor_density > bound) & ~sel
        else:
            sel_anchor = np.zeros(len(table), dtype=bool)
        for mask, flag_class in ((sel, "GATE"), (sel_anchor, "REPORT")):
            if not mask.any():
                continue
            frames.append(pd.DataFrame({
                "shape_id": table["shape_id"].to_numpy()[mask],
                "pooling_parent_id": table["pooling_parent_id"].to_numpy()[mask],
                "time_point": tp,
                "census": census[mask], "raked": raked[mask], "n_on": n_on[mask],
                "implied_density": implied[mask], "density_bound": bound[mask],
                "flag_class": flag_class,
            }))
    columns = ["shape_id", "pooling_parent_id", "time_point", "census", "raked", "n_on",
               "implied_density", "density_bound", "flag_class"]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return with_keys(out, "A2", unit, task_parent_id)


def unit_density(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> pd.DataFrame:
    """Per-unit density family inputs (port of audit_density_concentration.audit_table):
    anchor density, max density over the window, concentration and footprint-min ratio."""
    ctp = unit.census_time_point
    min_census = CHECKS["A3"].thresholds["min_census"]
    census, n_on_tc = _col(table, "census_population"), _col(table, f"n_on_{ctp}")
    keep = (census >= min_census) & (n_on_tc > 0)
    sub = table.loc[keep]
    tps = written_time_points(table)
    raked, n_on = _stack(sub, "raked", tps), _stack(sub, "n_on", tps)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(n_on > 0, raked / n_on, np.nan)
        dens_anchor = census[keep] / n_on_tc[keep]
        dens_max = np.nanmax(density, axis=1) if len(sub) else np.array([])
        conc = dens_max / dens_anchor
        footprint_min_ratio = n_on.min(axis=1) / n_on_tc[keep] if len(sub) else np.array([])
    return pd.DataFrame({
        "shape_id": sub["shape_id"].to_numpy(),
        "pooling_parent_id": sub["pooling_parent_id"].to_numpy(),
        "task_parent_id": task_parent_id,
        "census": census[keep], "n_on_tc": n_on_tc[keep],
        "dens_anchor": dens_anchor, "dens_max": dens_max, "conc": conc,
        "footprint_min_ratio": footprint_min_ratio,
    })


def check_a3_a4(density: pd.DataFrame, unit: CensusUnit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Census-level A3 (absolute + within-parent percentile), A4 (concentration) and the
    per-parent distribution artifact. Percentiles need the whole parent family, hence census level."""
    a3thr, a4thr = CHECKS["A3"].thresholds, CHECKS["A4"].thresholds
    dist_cols = ["pooling_parent_id", "n", "dens_q50", "dens_q99", "dens_max", "conc_q99", "conc_max",
                 "footprint_min_ratio_min", "flag_class"]
    if density.empty:
        empty = density.assign(family_n=pd.Series(dtype=int), family_median=pd.Series(dtype=float), flag_class=pd.Series(dtype=str))
        return (with_keys(empty.assign(flag_abs=pd.Series(dtype=bool), flag_rel=pd.Series(dtype=bool)), "A3", unit),
                with_keys(empty, "A4", unit),
                with_keys(pd.DataFrame(columns=dist_cols), "A3_dist", unit))
    density = density.copy()
    family = density.groupby("pooling_parent_id")["dens_anchor"]
    density["family_n"] = family.transform("size")
    density["family_median"] = family.transform("median")
    flag_abs = (density["dens_anchor"] > a3thr["anchor_abs"]).to_numpy()
    flag_rel = (
        (density["family_n"] >= MIN_FAMILY_UNITS)
        & (density["dens_anchor"] > a3thr["parent_median_mult"] * density["family_median"])
    ).to_numpy()
    a3 = density.loc[flag_abs | flag_rel].copy()
    a3["flag_abs"] = flag_abs[flag_abs | flag_rel]
    a3["flag_rel"] = flag_rel[flag_abs | flag_rel]
    a3["flag_class"] = "FLAG"
    a4 = density.loc[(density["conc"] > a4thr["conc"]).to_numpy()].copy()
    a4["flag_class"] = "FLAG"
    dist = (
        density.groupby("pooling_parent_id")
        .agg(
            n=("dens_anchor", "size"),
            dens_q50=("dens_anchor", "median"),
            dens_q99=("dens_anchor", lambda s: s.quantile(0.99)),
            dens_max=("dens_anchor", "max"),
            conc_q99=("conc", lambda s: s.quantile(0.99)),
            conc_max=("conc", "max"),
            footprint_min_ratio_min=("footprint_min_ratio", "min"),
        )
        .reset_index()
        .assign(flag_class="REPORT")
    )
    return with_keys(a3, "A3", unit), with_keys(a4, "A4", unit), with_keys(dist, "A3_dist", unit)


def check_a5_trajectory(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-unit swing (max/min raked over positive quarters), gate blinks (off quarters strictly
    between on quarters) and the largest one-quarter step. Returns (flagged units, task summary)."""
    thr = CHECKS["A5"].thresholds
    tps = written_time_points(table)
    raked, n_on = _stack(table, "raked", tps), _stack(table, "n_on", tps)
    on = n_on > 0
    any_on = on.any(axis=1)
    idx = np.arange(len(tps))[None, :]
    first_on = on.argmax(axis=1)
    last_on = len(tps) - 1 - on[:, ::-1].argmax(axis=1)
    between = (idx > first_on[:, None]) & (idx < last_on[:, None])
    blinks = np.where(any_on, ((~on) & between).sum(axis=1), 0)
    positive = raked > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rmax = np.where(positive, raked, -np.inf).max(axis=1)
        rmin = np.where(positive, raked, np.inf).min(axis=1)
        swing = np.where(positive.sum(axis=1) >= MIN_QUARTERS_FOR_SWING, rmax / rmin, 1.0)
        a, b = raked[:, :-1], raked[:, 1:]
        both = (a > 0) & (b > 0)
        step = np.where(both, np.maximum(a / b, b / a), 1.0).max(axis=1) if len(tps) > 1 else np.ones(len(table))
    flagged = (swing > thr["swing"]) | (blinks > thr["max_blinks"])
    census = _col(table, "census_population")
    rows = pd.DataFrame({
        "shape_id": table["shape_id"].to_numpy()[flagged],
        "pooling_parent_id": table["pooling_parent_id"].to_numpy()[flagged],
        "census": census[flagged], "swing": swing[flagged], "blinks": blinks[flagged],
        "max_step": step[flagged], "quarters_on": on.sum(axis=1)[flagged], "flag_class": "FLAG",
    })
    summary = pd.DataFrame([{
        "n_units": len(table), "n_flagged": int(flagged.sum()),
        "n_swing": int((swing > thr["swing"]).sum()), "n_blink": int((blinks > thr["max_blinks"]).sum()),
        "people_flagged": float(census[flagged].sum()), "flag_class": "REPORT",
    }])
    return with_keys(rows, "A5", unit, task_parent_id), with_keys(summary, "A5_summary", unit, task_parent_id)


def check_a6_gate_loss(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> pd.DataFrame:
    """Census people in units with no ON pixel: never within the window, and at the anchor."""
    tps = written_time_points(table)
    census, base = _col(table, "census_population"), _col(table, "base_population")
    never = (census > 0) & (_stack(table, "n_on", tps).sum(axis=1) == 0)
    anchor = (census > 0) & (base == 0)
    row = {
        "n_units_never_gated": int(never.sum()), "people_never_gated": float(census[never].sum()),
        "n_units_ungated_anchor": int(anchor.sum()), "people_ungated_anchor": float(census[anchor].sum()),
        "flag_class": "REPORT",
    }
    return with_keys(pd.DataFrame([row]), "A6", unit, task_parent_id)


def check_a7_capped_credit(table: pd.DataFrame, unit: CensusUnit, task_parent_id: str) -> pd.DataFrame:
    """Per time point: units capped, credit removed by the cap, credit applied, share removed."""
    ctp = unit.census_time_point
    census = _col(table, "census_population")
    rows = []
    for tp in written_time_points(table):
        if tp < ctp:
            continue
        removed = np.maximum(uncapped_credit(table, tp, ctp), 0.0)
        applied = np.maximum(_col(table, f"raked_{tp}") - census, 0.0)
        capped = removed > 1e-9 * np.maximum(census, 1.0)
        total = float(removed.sum() + applied.sum())
        rows.append({
            "time_point": tp, "n_units_capped": int(capped.sum()),
            "credit_removed": float(removed.sum()), "credit_applied": float(applied.sum()),
            "share_removed": float(removed.sum() / total) if total > 0 else 0.0, "flag_class": "REPORT",
        })
    return with_keys(pd.DataFrame(rows), "A7", unit, task_parent_id)


def check_a11_phantom_stock(
    table: pd.DataFrame,
    unit: CensusUnit,
    task_parent_id: str,
    names: pd.Series[str],
    parent_mass: pd.Series[float],
    rf_country: float,
) -> pd.DataFrame:
    """Census-0 units (C below the floor) with base_population > 0: per-unit list, post-anchor
    credit, and base as a share of the parent's pooled mass M_p (the bias on RF_p). FLAG when the
    phantom stock in people (base x rf_country) reaches the floor; smaller units are REPORT rows,
    since fine-unit censuses (AUS mesh blocks, JPN small areas) have 10^5 census-0 units."""
    floor = CHECKS["A11"].thresholds["census_zero_floor"]
    ctp, last = unit.census_time_point, written_time_points(table)[-1]
    sel = (table["census_population"] < floor) & (table["base_population"] > 0)
    out = table.loc[sel, ["shape_id", "pooling_parent_id", "census_population", "base_population",
                          f"n_on_{ctp}", f"raked_{last}"]].copy()
    out = out.rename(columns={f"n_on_{ctp}": "n_on_tc", f"raked_{last}": "raked_last"})
    out["shape_name"] = out["shape_id"].map(names)
    out["post_anchor_credit"] = out["raked_last"] - out["census_population"]
    out["base_share_of_parent_mass"] = out["base_population"] / out["pooling_parent_id"].map(parent_mass)
    out["phantom_people"] = out["base_population"] * rf_country
    out["flag_class"] = np.where(out["phantom_people"] >= floor, "FLAG", "REPORT")
    return with_keys(out, "A11", unit, task_parent_id)


# ---------------------------------------------------------------- census-level checks


def check_a8_a10(rf_prior: pd.DataFrame, unit: CensusUnit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A8: rf_country / lambda-probe roll-up (the relative band is applied across censuses by the
    collator). A10: parents that are prior-dominated AND uncapped -- the census_rf_main warning
    condition, verbatim."""
    thr10 = CHECKS["A10"].thresholds
    rf_c = float(rf_prior["rf_country"].iloc[0])
    a8 = pd.DataFrame([{
        "rf_country": rf_c,
        "lambda_empirical": float(rf_prior["lambda_empirical"].iloc[0]),
        "lambda_probe_status": str(rf_prior["lambda_probe_status"].iloc[0]),
        "n_parents": len(rf_prior),
        "census_total": float(rf_prior["census_population"].sum()),
        "predicted_total": float(rf_prior["predicted_population"].sum()),
        "flag_class": "REPORT",
    }])
    if rf_c > 0:
        prior_dominated = rf_prior["predicted_population"] < thr10["lambda_persons"] / rf_c
        trigger = prior_dominated & rf_prior["density_bound"].isna() & (rf_prior["census_population"] > 0)
    else:
        trigger = pd.Series(data=False, index=rf_prior.index)
    a10 = rf_prior.loc[trigger, ["pooling_parent_id", "census_population", "predicted_population", "pooling_level"]].copy()
    a10["flag_class"] = "FLAG"
    return with_keys(a8, "A8", unit), with_keys(a10, "A10", unit)


def check_a9_vanished(finest: pd.DataFrame, seen: set[str], unit: CensusUnit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerated finest units that appear in no table (zero covered pixels at every time point)."""
    enumerated = finest["population_total"].notna()
    vanished = finest.loc[enumerated & ~finest["shape_id"].isin(seen), ["shape_id", "shape_name", "population_total"]]
    rows = vanished.rename(columns={"population_total": "people"})
    rows["flag_class"] = np.where(rows["people"] > 0, "FLAG", "REPORT")
    summary = pd.DataFrame([{
        "n_finest": len(finest), "n_in_tables": int(finest["shape_id"].isin(seen).sum()),
        "n_vanished": len(vanished), "people_vanished": float(vanished["population_total"].sum()),
        "n_unenumerated": int((~enumerated).sum()), "flag_class": "REPORT",
    }])
    return with_keys(rows, "A9", unit), with_keys(summary, "A9_summary", unit)


def unmodeled_shape_ids(
    pm_data: PopulationModelData, model_spec: ModelSpecification, unit: CensusUnit, max_level: int
) -> set[str]:
    """Finest census units whose representative point falls in a GBD location with a non-finite
    final raking factor at the census time point (unmodeled / zero-population supplement
    locations: their census-raked pixels are wiped to NaN in the product)."""
    ctp = unit.census_time_point
    rf_path = pm_data.raking_factor_path(ctp, model_spec, stage="final")
    if not rf_path.exists():
        print(f"no final raking factors at {ctp}; skipping the unmodeled-location class")
        return set()
    rf = pd.read_parquet(rf_path, columns=["location_id", "raking_factor"])
    bad = rf.loc[~np.isfinite(rf["raking_factor"]), "location_id"].unique()
    if len(bad) == 0:
        return set()
    crs = pm_data.load_modeling_frame_info(model_spec.resolution).crs
    shapes = pm_data.load_raking_shapes(version=RAKING_VERSION).to_crs(crs)
    unmodeled = shapes[shapes["location_id"].isin(bad)]
    task_admins, _ = pm_data.load_census_raking_inputs(model_spec, iso3=unit.iso3, census_time_point=ctp)
    task_admins = task_admins.set_crs(crs, allow_override=True).reset_index()
    # spatial-index joins throughout: no polygon unions (a union of a coastline-heavy country's
    # task admins takes minutes, and testing every unit against it took hours for AUS/JPN/USA)
    touched = gpd.sjoin(unmodeled[["location_id", "geometry"]], task_admins[["geometry"]], predicate="intersects", how="inner")
    hit = unmodeled[unmodeled["location_id"].isin(touched["location_id"])]
    if hit.empty:
        return set()
    modeled = shapes[~shapes["location_id"].isin(bad)]
    result: set[str] = set()
    for _, shape in hit.iterrows():
        units = gpd.read_parquet(
            pm_data.census_path(unit.iso3, census_year(ctp)),
            columns=["shape_id", "admin_level", "geometry"],
            bbox=tuple(shape.geometry.bounds),
            filters=[("admin_level", "==", max_level)],
        )
        if units.empty:
            continue
        units = units.set_crs(crs, allow_override=True)
        one = gpd.GeoDataFrame({"location_id": [shape["location_id"]]}, geometry=[shape.geometry], crs=crs)
        # a unit belongs to the unmodeled location when its representative point falls inside it,
        # or (archipelago units whose point lands in water) when its polygon touches the unmodeled
        # location and no modeled one
        points = units.set_geometry(units.representative_point())
        inside = set(gpd.sjoin(points, one, predicate="within", how="inner")["shape_id"])
        touching = units[units["shape_id"].isin(gpd.sjoin(units, one, predicate="intersects", how="inner")["shape_id"])]
        near_modeled = modeled[modeled.intersects(shapely.box(*shape.geometry.bounds))]
        if len(near_modeled) and len(touching):
            touches_modeled = set(gpd.sjoin(touching, near_modeled[["geometry"]], predicate="intersects", how="inner")["shape_id"])
        else:
            touches_modeled = set()
        result |= inside | (set(touching["shape_id"]) - touches_modeled)
    return result


def check_a12_zero_mismatch(
    values: pd.DataFrame, unmodeled: set[str], a3: pd.DataFrame, a11: pd.DataFrame, a9: pd.DataFrame, unit: CensusUnit
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every off-axis point of the census-vs-prediction fit plots, labeled by cause:
    (a) census>0 with no ON pixel at t_c (in tables) or vanished (a_vanished);
    (b) census>0 in an unmodeled GBD location; (c) census>0 with extreme smearing (A3 absolute);
    (d) census-0 with model stock (A11). Returns (unit rows, per-class summary)."""
    v = values
    classes = {
        "a_ungated": (v["census"] > 0) & (v["n_on_tc"] == 0),
        "b_unmodeled": (v["census"] > 0) & v["shape_id"].isin(unmodeled),
        "c_smearing": v["shape_id"].isin(a3.loc[a3["flag_abs"], "shape_id"]) if len(a3) else pd.Series(data=False, index=v.index),
        "d_phantom": v["shape_id"].isin(a11["shape_id"]),
    }
    frames = [
        v.loc[mask, ["shape_id", "pooling_parent_id", "census", "base", "n_on_tc"]].assign(mismatch_class=name)
        for name, mask in classes.items()
    ]
    frames.append(pd.DataFrame({
        "shape_id": a9["shape_id"].to_numpy(), "pooling_parent_id": None, "census": a9["people"].to_numpy(),
        "base": 0.0, "n_on_tc": 0, "mismatch_class": "a_vanished",
    }))
    rows = pd.concat(frames, ignore_index=True).assign(flag_class="REPORT")
    summary = (
        rows.groupby("mismatch_class").agg(n_units=("shape_id", "size"), people=("census", "sum"))
        .reindex(["a_ungated", "a_vanished", "b_unmodeled", "c_smearing", "d_phantom"], fill_value=0)
        .reset_index().assign(flag_class="REPORT")
    )
    return with_keys(rows, "A12", unit), with_keys(summary, "A12_summary", unit)


# ---------------------------------------------------------------- task entry point


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, keeping the first frame's columns when every part is empty."""
    non_empty = [part for part in parts if len(part)]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)
    return parts[0].iloc[0:0].copy() if parts else pd.DataFrame()


class TaskResults(NamedTuple):
    checks: dict[str, pd.DataFrame]
    density: pd.DataFrame
    unit_values: pd.DataFrame
    totals: pd.DataFrame
    n_tables: int
    n_missing: int


def run_task_checks(
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
    unit: CensusUnit,
    task_parent_ids: pd.Series[str],
    names: pd.Series[str],
    parent_mass: pd.Series[float],
    rf_country: float,
) -> TaskResults:
    """Per-task checks over every populated table of one census, concatenated per check."""
    ctp = unit.census_time_point
    out: dict[str, list[pd.DataFrame]] = defaultdict(list)
    density_parts: list[pd.DataFrame] = []
    value_parts: list[pd.DataFrame] = []
    total_parts: list[pd.DataFrame] = []
    n_missing = 0
    for task_parent_id in task_parent_ids:
        try:
            table = pm_data.load_raked_census_table(unit.iso3, task_parent_id, ctp, model_spec)
        except FileNotFoundError:
            n_missing += 1  # unpopulated tasks write no table
            continue
        if table.empty:
            continue
        out["A1"].append(check_a1_anchor(table, unit, task_parent_id))
        out["A2"].append(check_a2_growth_density(table, unit, task_parent_id))
        a5, a5_summary = check_a5_trajectory(table, unit, task_parent_id)
        out["A5"].append(a5)
        out["A5_summary"].append(a5_summary)
        out["A6"].append(check_a6_gate_loss(table, unit, task_parent_id))
        out["A7"].append(check_a7_capped_credit(table, unit, task_parent_id))
        out["A11"].append(check_a11_phantom_stock(table, unit, task_parent_id, names, parent_mass, rf_country))
        density_parts.append(unit_density(table, unit, task_parent_id))
        value_parts.append(
            table[["shape_id", "pooling_parent_id", "census_population", "base_population", f"n_on_{ctp}", f"raked_{ctp}"]]
            .rename(columns={"census_population": "census", "base_population": "base", f"n_on_{ctp}": "n_on_tc", f"raked_{ctp}": "raked_tc"})
        )
        raked_cols = [c for c in table.columns if c.startswith("raked_")]
        total_parts.append(table[["pooling_parent_id", "census_population", *raked_cols]])
    values = _concat(value_parts)
    if values.empty:
        values = pd.DataFrame(columns=["shape_id", "pooling_parent_id", "census", "base", "n_on_tc", "raked_tc"])
    totals = _concat(total_parts)
    if not totals.empty:
        totals = totals.groupby("pooling_parent_id").sum(min_count=1).rename(columns={"census_population": "census"}).reset_index()
    return TaskResults({k: _concat(v) for k, v in out.items()}, _concat(density_parts), values, totals, len(out["A1"]), n_missing)


def census_main(iso3: str, census_time_point: str, resolution: str, version: str, output_dir: str) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    unit = CensusUnit(iso3, census_time_point)
    ctp = census_time_point

    tasks = pm_data.load_census_raking_tasks(model_spec)
    tasks = tasks[(tasks["iso3"] == iso3) & (tasks["census_time_point"] == ctp)]
    rf_prior = pm_data.load_census_rf_prior(iso3, ctp, model_spec)
    units = pd.read_parquet(pm_data.census_path(iso3, census_year(ctp)), columns=UNIT_COLUMNS)
    max_level = int(units["admin_level"].max())
    finest = units[units["admin_level"] == max_level]
    names = finest.set_index("shape_id")["shape_name"]
    parent_mass = rf_prior.set_index("pooling_parent_id")["predicted_population"]

    print(f"{iso3} {ctp}: {len(tasks)} tasks")
    rf_country = float(rf_prior["rf_country"].iloc[0])
    task = run_task_checks(pm_data, model_spec, unit, tasks["task_parent_id"], names, parent_mass, rf_country)
    results = dict(task.checks)
    a3, a4, a3_dist = check_a3_a4(task.density, unit)
    results.update({"A3": a3, "A4": a4, "A3_dist": a3_dist})
    results["A8"], results["A10"] = check_a8_a10(rf_prior, unit)
    a9, a9_summary = check_a9_vanished(finest, set(task.unit_values["shape_id"]), unit)
    results.update({"A9": a9, "A9_summary": a9_summary})
    unmodeled = unmodeled_shape_ids(pm_data, model_spec, unit, max_level)
    a11 = results.get("A11", pd.DataFrame(columns=["shape_id"]))
    results["A12"], results["A12_summary"] = check_a12_zero_mismatch(task.unit_values, unmodeled, a3, a11, a9, unit)
    results["C2"] = with_keys(lane_c.fidelity_by_level(units, task.unit_values), "C2", unit)  # base (model at t_c) vs census
    results["A_parent_totals"] = with_keys(task.totals, "A_parent_totals", unit)

    for check_id, frame in results.items():
        artifacts.save_check(pm_data, frame, resolution, version, check_id, iso3, ctp)
    gates = int((results["A1"]["flag_class"] == "GATE").sum()) if "A1" in results and len(results["A1"]) else 0
    print(
        f"{iso3} {ctp}: tables {task.n_tables} (missing {task.n_missing}); A1 gate failures {gates}; "
        f"A11 phantom units {len(a11)} ({int((a11['flag_class'] == 'FLAG').sum()) if len(a11) else 0} flagged); "
        f"A9 vanished {len(a9)}; A12 unmodeled {len(unmodeled)}"
    )
