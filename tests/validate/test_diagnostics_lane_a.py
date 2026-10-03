import numpy as np
import pandas as pd
import pytest

from rra_population_model.validate.diagnostics import lane_a
from rra_population_model.validate.diagnostics.lane_a import CensusUnit

UNIT = CensusUnit("XXX", "2022q1")
TPS = ["2021q4", "2022q1", "2022q2"]


def make_table() -> pd.DataFrame:
    """Four units: an established unit, a growing unit, a census-0 unit with model stock, an ungated unit.

    Columns mirror census_rake.utils.compute_shape_values; raked values obey the v3 formula
    y = C + rf * built * built / (built + base), so the checks see a consistent table."""
    census = np.array([100.0, 50.0, 0.0, 20.0])
    base = np.array([10.0, 5.0, 2.0, 0.0])
    rf = np.array([10.0] * 4)
    bound = np.array([5.0, 5.0, 5.0, np.nan])
    table = pd.DataFrame({
        "shape_id": ["u1", "u2", "u3", "u4"],
        "pooling_parent_id": ["p"] * 4,
        "census_population": census,
        "base_population": base,
        "rf_parent": rf,
        "density_bound": bound,
    })
    built = {"2021q4": np.zeros(4), "2022q1": np.zeros(4), "2022q2": np.array([0.0, 2.0, 1.0, 0.0])}
    n_on = {"2021q4": [20, 10, 4, 0], "2022q1": [20, 10, 4, 0], "2022q2": [20, 12, 5, 0]}
    for tp in TPS:
        predicted = base + built[tp]
        weight = np.where(built[tp] + base > 0, built[tp] / np.maximum(built[tp] + base, 1e-300), 0.0)
        credit = rf * built[tp] * weight if tp >= "2022q1" else 0.0
        table[f"shape_population_{tp}"] = predicted
        table[f"built_{tp}"] = built[tp]
        table[f"raked_{tp}"] = np.where(predicted > 0, census + credit, 0.0)
        table[f"n_on_{tp}"] = n_on[tp]
    return table


def test_a1_passes_then_fails_when_anchor_perturbed() -> None:
    table = make_table()
    assert lane_a.check_a1_anchor(table, UNIT, "t")["flag_class"].item() == "REPORT"
    table.loc[0, "raked_2022q1"] += 1.0
    assert lane_a.check_a1_anchor(table, UNIT, "t")["flag_class"].item() == "GATE"


def test_a2_gates_only_credited_exceedance() -> None:
    table = make_table()
    # u2 at 2022q2: raked 50 + 10*2*2/7 ~ 55.7 over 12 pixels = 4.6 < bound 5 -> nothing
    assert lane_a.check_a2_growth_density(table, UNIT, "t").empty
    table.loc[1, "raked_2022q2"] = 80.0  # credited and above the bound: cap regression
    out = lane_a.check_a2_growth_density(table, UNIT, "t")
    assert out["flag_class"].tolist() == ["GATE"]
    # an established unit denser than its bound is the accepted at-anchor extreme: listed, not flagged
    table.loc[0, "density_bound"] = 1.0
    out = lane_a.check_a2_growth_density(table, UNIT, "t")
    assert set(out["flag_class"]) == {"GATE", "REPORT"}
    assert out.loc[out["flag_class"] == "REPORT", "shape_id"].item() == "u1"


def test_uncapped_credit_matches_the_table() -> None:
    table = make_table()
    removed = lane_a.uncapped_credit(table, "2022q2", "2022q1")
    assert np.allclose(removed, 0.0)  # nothing capped in the fixture
    assert np.allclose(lane_a.uncapped_credit(table, "2021q4", "2022q1"), 0.0)  # pre-anchor: no credit


def test_a11_lists_census_zero_units_with_stock() -> None:
    table = make_table()
    names = pd.Series({"u3": "Forest"})
    mass = pd.Series({"p": 17.0})
    out = lane_a.check_a11_phantom_stock(table, UNIT, "t", names, mass, rf_country=10.0)
    assert out["shape_id"].tolist() == ["u3"]
    assert out["flag_class"].item() == "FLAG"  # 2 base x 10 = 20 phantom people, above the floor
    small = lane_a.check_a11_phantom_stock(table, UNIT, "t", names, mass, rf_country=1.0)
    assert small["flag_class"].item() == "REPORT"
    assert out["shape_name"].item() == "Forest"
    assert out["post_anchor_credit"].item() == pytest.approx(table.loc[2, "raked_2022q2"])
    assert out["base_share_of_parent_mass"].item() == pytest.approx(table.loc[2, "base_population"] / mass["p"])


def test_a5_flags_blink_and_swing() -> None:
    table = make_table()
    table.loc[1, "n_on_2022q1"] = 0  # u2 blinks between two on quarters
    table.loc[0, "raked_2022q2"] = 300.0  # u1 swings 3x
    rows, summary = lane_a.check_a5_trajectory(table, UNIT, "t")
    by_unit = rows.set_index("shape_id")
    assert set(by_unit.index) == {"u1", "u2"}
    assert by_unit.loc["u2", "blinks"] == 1
    assert by_unit.loc["u1", "swing"] == pytest.approx(table.loc[0, "raked_2022q2"] / table.loc[0, "raked_2021q4"])
    assert summary["n_flagged"].item() == len(by_unit)


def test_a6_and_a12_classify_the_ungated_unit() -> None:
    table = make_table()
    a6 = lane_a.check_a6_gate_loss(table, UNIT, "t")
    assert a6["n_units_never_gated"].item() == 1
    assert a6["people_never_gated"].item() == table.loc[3, "census_population"]
    values = table[["shape_id", "pooling_parent_id", "census_population", "base_population", "n_on_2022q1", "raked_2022q1"]].rename(
        columns={"census_population": "census", "base_population": "base", "n_on_2022q1": "n_on_tc", "raked_2022q1": "raked_tc"}
    )
    a3 = pd.DataFrame(columns=["shape_id", "flag_abs"])
    a11 = pd.DataFrame({"shape_id": ["u3"]})
    a9 = pd.DataFrame({"shape_id": ["u9"], "people": [7.0]})
    rows, summary = lane_a.check_a12_zero_mismatch(values, {"u1"}, a3, a11, a9, UNIT)
    by_class = summary.set_index("mismatch_class")["n_units"].to_dict()
    assert by_class == {"a_ungated": 1, "a_vanished": 1, "b_unmodeled": 1, "c_smearing": 0, "d_phantom": 1}
    assert summary.set_index("mismatch_class").loc["b_unmodeled", "people"] == table.loc[0, "census_population"]


def test_a8_band_and_a10_shared_trigger() -> None:
    rf_prior = pd.DataFrame({
        "pooling_parent_id": ["p1", "p2"],
        "census_population": [1000.0, 500.0],
        "predicted_population": [900.0, 1.0],
        "pooling_level": [2, 2],
        "density_bound": [4.0, np.nan],
        "rf_country": [1.1, 1.1],
        "lambda_empirical": [800.0, 800.0],
        "lambda_probe_status": ["informational"] * 2,
    })
    a8, a10 = lane_a.check_a8_a10(rf_prior, UNIT)
    assert a8["rf_country"].item() == pytest.approx(1.1)
    assert a10["pooling_parent_id"].tolist() == ["p2"]  # prior-dominated (1 < 800/1.1) and uncapped


def test_a9_vanished_units() -> None:
    finest = pd.DataFrame({"shape_id": ["u1", "u2", "u3"], "shape_name": list("abc"), "population_total": [5.0, np.nan, 9.0]})
    rows, summary = lane_a.check_a9_vanished(finest, {"u1"}, UNIT)
    assert rows["shape_id"].tolist() == ["u3"]  # u2 is unenumerated (NaN), dropped by design
    assert rows["flag_class"].item() == "FLAG"  # carries people; a zero-population water unit would be REPORT
    assert summary["n_unenumerated"].item() == 1
    assert summary["people_vanished"].item() == finest.loc[2, "population_total"]
