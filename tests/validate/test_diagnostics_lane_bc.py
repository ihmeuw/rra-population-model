import numpy as np
import pandas as pd
import pytest

from rra_population_model.validate.diagnostics import collate, lane_c
from rra_population_model.validate.diagnostics.lane_b import _trajectory_stats
from rra_population_model.validate.diagnostics.registry import CHECKS


def test_trajectory_stats_detect_swing_blink_step() -> None:
    # 4 quarters, 1x3 pixels: steady / blinking / swinging 3x in one step
    stack = np.array([
        [[10.0, 5.0, 10.0]],
        [[10.0, 0.0, 30.0]],
        [[10.0, 5.0, 30.0]],
        [[10.0, 5.0, 30.0]],
    ])
    stats = _trajectory_stats(stack, swing_thr=2.0, blink_thr=0)
    expected = {"n_pixels_ever_positive": 3, "n_blink": 1, "n_swing": 1, "n_step": 1, "people_swing_last": 30.0}
    assert {k: stats[k] for k in expected} == expected


def test_held_out_scores_use_nearest_written_quarter() -> None:
    totals = pd.DataFrame([
        {"iso3": "X", "census_time_point": "2021q1", "pooling_parent_id": "p", "census": 100.0, "raked_2021q1": 100.0, "raked_2021q4": 110.0},
        {"iso3": "X", "census_time_point": "2022q1", "pooling_parent_id": "p", "census": 120.0, "raked_2022q1": 120.0},
    ])
    out = lane_c.held_out_scores(totals)
    row = out.set_index(["anchor_census", "target_census"]).loc[("2021q1", "2022q1")]
    assert (row["time_point_used"], row["offset_quarters"]) == ("2021q4", -1)
    assert (row["wmape_flat"], row["wmape_v3"]) == pytest.approx((20 / 120, 10 / 120))
    # the 2022 census has no written quarter near 2021q1 -> no reverse row
    assert ("2022q1", "2021q1") not in out.set_index(["anchor_census", "target_census"]).index


def test_fidelity_by_level_scales_nationally() -> None:
    units = pd.DataFrame({
        "shape_id": ["A", "A1", "A2"], "parent_id": [None, "A", "A"], "admin_level": [0, 1, 1],
        "shape_name": ["a", "a1", "a2"], "path_to_top_parent": ["A", "A,A1", "A,A2"], "population_total": [30.0, 10.0, 20.0],
    })
    values = pd.DataFrame({"shape_id": ["A1", "A2"], "census": [10.0, 20.0], "base": [1.0, 1.0]})
    out = lane_c.fidelity_by_level(units, values).set_index("admin_level")
    assert out.loc[0, "rmse_log10"] == pytest.approx(0.0)  # national total matches by construction
    assert out.loc[1, "n_units"] == len(values)
    assert out.loc[1, "rmse_log10"] > 0
    assert out.loc[1, "national_scale"] == pytest.approx(values["census"].sum() / values["base"].sum())


def test_fidelity_rejects_unordered_paths() -> None:
    units = pd.DataFrame({
        "shape_id": ["A", "A1"], "admin_level": [0, 1], "path_to_top_parent": ["A", "A1,A"], "population_total": [1.0, 1.0],
    })
    with pytest.raises(ValueError, match="not ordered"):
        lane_c.check_path_order(units)


def test_summarize_and_markdown() -> None:
    check = CHECKS["A1"]
    df = pd.DataFrame({"flag_class": ["REPORT", "GATE"], "abs_error": [0.0, 3.0]})
    s = collate.summarize(check, df, "checks/A1")
    assert s.gate_failed
    assert (s.n_gate, s.headline["max_abs_error"]) == (1, df["abs_error"].max())
    flag_only = collate.summarize(CHECKS["A11"], pd.DataFrame({"flag_class": ["FLAG"], "post_anchor_credit": [2.0]}), "x")
    assert not flag_only.gate_failed
    assert flag_only.n_flagged == 1
    md = collate.render_markdown([s, flag_only], "v", None, gate_failed=True)
    assert "**FAIL**" in md
    assert "FLAG 1" in md
    assert "Release gate: FAILED" in md


def test_b3_flags_jump_at_weight_transition() -> None:
    scan = pd.DataFrame({
        "time_point": ["2021q4", "2022q1", "2021q4", "2022q1"],
        "location_id": [1, 1, 2, 2],
        "sum_final": [1000.0, 1200.0, 1000.0, 1010.0],
    })
    weights = pd.DataFrame({
        "iso3": ["X"] * 3, "model_time_point": ["2021q4", "2021q4", "2022q1"],
        "census_time_point": ["2021q1", "2022q1", "2022q1"], "weight": [0.5, 0.5, 1.0],
    }).set_index(["iso3", "model_time_point", "census_time_point"])
    loc_iso3 = pd.Series({1: "X", 2: "X"})
    out = collate.b3_splice_smoothness(scan, weights, loc_iso3).set_index("location_id")
    assert out["flag_class"].to_dict() == {1: "FLAG", 2: "REPORT"}


def test_b4_rolls_up_unmodeled_class() -> None:
    a12 = pd.DataFrame({
        "iso3": ["FIN"] * 3, "census_time_point": ["2022q1"] * 3, "shape_id": ["a", "b", "c"],
        "census": [10.0, 20.0, 5.0], "mismatch_class": ["b_unmodeled", "b_unmodeled", "d_phantom"],
    })
    out = collate.b4_unmodeled(a12)
    unmodeled = a12[a12["mismatch_class"] == "b_unmodeled"]
    assert (out["n_units"].item(), out["people"].item()) == (len(unmodeled), unmodeled["census"].sum())


def test_a8_relative_band_flags_outliers_only() -> None:
    a8 = pd.DataFrame({"iso3": ["A", "B", "C", "D"], "rf_country": [1500.0, 1700.0, 1600.0, 80000.0]})
    out = collate.a8_relative_band(a8)
    assert out["flag_class"].tolist() == ["REPORT", "REPORT", "REPORT", "FLAG"]


def test_summarize_reports_per_time_point_and_universe() -> None:
    df = pd.DataFrame({"flag_class": ["FLAG", "REPORT"], "time_point": ["2020q1", "2020q2"], "people": [10.0, 30.0]})
    s = collate.summarize(CHECKS["B4"], df, "x", universe=(4, "things"))
    assert s.headline["people/tp"] == pytest.approx(20.0)
    assert (s.universe, s.universe_label) == (4, "things")
    md = collate.render_markdown([s], "v", None, gate_failed=False)
    assert "1 of 4 things (25.00%)" in md


def test_b1_reclassify_uses_shares() -> None:
    scan = pd.DataFrame({
        "flag_class": ["FLAG", "FLAG", "GATE"], "n_finite": [1_000_000, 1_000, 10],
        "n_over_ppp_flag": [5, 5, 0], "n_over_ppm3": [0, 0, 0], "n_pop_no_volume": [0, 0, 0], "n_over_ppp_list": [0, 0, 0],
    })
    out = collate.b1_reclassify(scan)
    assert out["flag_class"].tolist() == ["REPORT", "FLAG", "GATE"]  # 5e-6 share, 5e-3 share, GATE kept


def test_b1_implausible_labels_mechanisms() -> None:
    pixels = pd.DataFrame({
        "check_id": ["B1_pixels"] * 3,  # the scan stamps its own id; the view must replace it
        "time_point": ["2020q1"] * 3, "block_key": ["b"] * 3, "x": [0.0, 40.0, 80.0], "y": [0.0, 0.0, 0.0],
        "value": [3000.0, 1500.0, 900.0], "raw": [0.001, 0.1, 1.0], "volume_m3": [50.0, 40000.0, 40000.0], "location_id": [1, 1, 1],
    })
    attr = pd.DataFrame({
        "time_point": ["2020q1"] * 2, "block_key": ["b"] * 2, "x": [0.0, 40.0], "y": [0.0, 0.0],
        "rf1": [1000.0, 1000.0], "rf2": [1.0, 1.0], "census_layer": [3000.0, 1500.0], "stage": ["census", "census"],
        "census_iso3": ["USA", "USA"], "census_time_point": ["2020q1", "2020q1"],
    })
    out = collate.b1_implausible(pixels, attr, pd.Series({1: "Somewhere"}), "EPSG:3857")
    assert out["mechanism"].tolist() == ["census mass on low-volume pixel", "census amplifies model"]
    assert out["physically_impossible"].tolist() == [True, False]
    assert out["flag_class"].tolist() == ["FLAG", "REPORT"]
