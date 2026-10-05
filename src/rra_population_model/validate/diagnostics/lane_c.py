"""Lane C: cross-release and external-consistency checks.

Parquet-only. C2 runs inside the Lane A census task (it needs one census's
tables and hierarchy); C1 and C3 run in the collator over Lane A artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rra_population_model.validate.diagnostics.registry import CHECKS

MIN_UNITS_FOR_CORRELATION = 3
MIN_CENSUSES = 2


def tp_to_quarters(time_point: str) -> int:
    year, quarter = time_point.split("q")
    return int(year) * 4 + int(quarter) - 1


def check_path_order(units: pd.DataFrame) -> None:
    """path_to_top_parent is comma-separated from the top parent down to the unit itself, so
    position == admin level. Verified on a sample against the file's own admin_level column;
    fails loudly if a census breaks the convention."""
    level_of = units.set_index("shape_id")["admin_level"]
    for parts in units["path_to_top_parent"].str.split(",").head(1000):
        for pos, sid in enumerate(parts):
            if sid in level_of.index and int(level_of[sid]) != pos:
                msg = f"path_to_top_parent is not ordered by admin level ({sid} at position {pos})"
                raise ValueError(msg)


def fidelity_by_level(units: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """C2: the model's anchor prediction (base = pred(t_c), nationally scaled to the census
    total -- the intercept-only frame of the fit plots) vs census, aggregated up every admin
    level. raked(t_c) is exact by A1, so the prediction is the informative side.

    ``units`` is the census file (all levels, no geometry); ``values`` the finest units seen
    in the tables with columns shape_id, census, base. Units absent from the tables count
    as predicted 0 with their census count.
    """
    max_level = int(units["admin_level"].max())
    finest = units.loc[units["admin_level"] == max_level, ["shape_id", "path_to_top_parent", "population_total"]]
    df = finest.merge(values[["shape_id", "census", "base"]], on="shape_id", how="left")
    df["census"] = df["census"].fillna(df["population_total"]).fillna(0.0)
    df["base"] = df["base"].fillna(0.0)
    scale = df["census"].sum() / df["base"].sum() if df["base"].sum() > 0 else np.nan
    df["raked_tc"] = df["base"] * scale  # scaled model prediction, named for the aggregation below
    check_path_order(units)
    path = df["path_to_top_parent"].str.split(",")
    rows = []
    for level in range(max_level + 1):
        key = path.str[level] if level < max_level else df["shape_id"]
        g = df.groupby(key)[["census", "raked_tc"]].sum()
        g = g.loc[pd.notna(g.index)]
        x = np.log10(g["census"].to_numpy(dtype=float) + 1)
        y = np.log10(g["raked_tc"].to_numpy(dtype=float) + 1)
        rmse = float(np.sqrt(np.mean((y - x) ** 2))) if len(g) else np.nan
        r = float(np.corrcoef(x, y)[0, 1]) if len(g) >= MIN_UNITS_FOR_CORRELATION and x.std() > 0 and y.std() > 0 else np.nan
        rows.append({
            "admin_level": level,
            "n_units": int(len(g)),
            "national_scale": float(scale),
            "rmse_log10": rmse,
            "r_log10": r,
            "n_census0_pred_pos": int(((g["census"] == 0) & (g["raked_tc"] > 0)).sum()),
            "n_census_pos_pred0": int(((g["census"] > 0) & (g["raked_tc"] == 0)).sum()),
            "flag_class": "REPORT",
        })
    return pd.DataFrame(rows)


def held_out_scores(parent_totals: pd.DataFrame) -> pd.DataFrame:
    """C3: for multi-census countries, flat (census_k) vs v3 (raked_k at the quarter nearest
    census j's date) against census j, by pooling parent, weighted MAPE.

    Tables never carry the other census's exact date (its weight is 0 there), so the
    nearest written quarter within ``max_offset_quarters`` is used and the offset recorded.
    """
    thr = CHECKS["C3"].thresholds
    columns = ["iso3", "anchor_census", "target_census", "time_point_used", "offset_quarters",
               "n_parents", "truth_people", "wmape_flat", "wmape_v3", "flag_class"]
    rows: list[dict[str, object]] = []
    if parent_totals.empty:
        return pd.DataFrame(columns=columns)
    for iso3, g in parent_totals.groupby("iso3"):
        censuses = sorted(g["census_time_point"].unique())
        if len(censuses) < MIN_CENSUSES:
            continue
        for k in censuses:
            gk = g[g["census_time_point"] == k].set_index("pooling_parent_id")
            written = {c[len("raked_"):]: c for c in gk.columns if c.startswith("raked_") and gk[c].notna().any()}
            for j in censuses:
                if j == k or not written:
                    continue
                target = tp_to_quarters(j)
                tp = min(written, key=lambda t: abs(tp_to_quarters(t) - target))
                offset = tp_to_quarters(tp) - target
                if abs(offset) > thr["max_offset_quarters"]:
                    continue
                gj = g[g["census_time_point"] == j].set_index("pooling_parent_id")["census"]
                both = gk.index.intersection(gj.index)
                truth = gj.loc[both].to_numpy(dtype=float)
                if len(both) == 0 or truth.sum() <= 0:
                    continue
                flat = gk.loc[both, "census"].to_numpy(dtype=float)
                v3 = gk.loc[both, written[tp]].to_numpy(dtype=float)
                rows.append({
                    "iso3": iso3, "anchor_census": k, "target_census": j, "time_point_used": tp,
                    "offset_quarters": int(offset), "n_parents": int(len(both)),
                    "truth_people": float(truth.sum()),
                    "wmape_flat": float(np.abs(flat - truth).sum() / truth.sum()),
                    "wmape_v3": float(np.abs(v3 - truth).sum() / truth.sum()),
                    "flag_class": "REPORT",
                })
    return pd.DataFrame(rows, columns=columns)


def cross_release(
    current: pd.DataFrame, previous: pd.DataFrame, keys: list[str], value: str
) -> pd.DataFrame:
    """C1: outer-join two releases on ``keys`` and rank by absolute change in ``value``."""
    merged = current.merge(previous, on=keys, how="outer", suffixes=("", "_prev"))
    merged["abs_diff"] = merged[value] - merged[f"{value}_prev"]
    prev = merged[f"{value}_prev"].replace(0, np.nan)
    merged["rel_diff"] = merged["abs_diff"] / prev
    merged["flag_class"] = "REPORT"
    order = merged["abs_diff"].abs().sort_values(ascending=False, na_position="last").index
    return merged.loc[order].reset_index(drop=True)
