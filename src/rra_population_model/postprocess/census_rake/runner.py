from pathlib import Path
from typing import Any

from loguru import logger
import functools

import click
from affine import Affine
from rasterio import features
from rra_tools import jobmon, parallel

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import ModelSpecification
from rra_population_model.postprocess.census_rake import utils

# 0 = don't downsample; 1+ = number of downsampled admin-years to run
DOWNSAMPLE_ADMINS = 0

# Task-splitting thresholds: compute sizing ONLY (memory ~ census cache +
# bbox + covered area; see the resource-model comment further down). As of
# 2026-08-28 the statistical structures are DECOUPLED from the task partition:
# RF pooling AND the density-cap families both use semantic pooling parents
# (one admin level above the census's most detailed, capped at admin2;
# national when admin1 is most detailed), computed by the census_rf pre-stage
# and looked up per unit -- so these thresholds can be retuned for compute
# freely. Pooling-scale sensitivity itself is measured flat
# (.claude/census_rake_temporal/pooling_sensitivity.py: one level finer
# through fully national, wMAPE spread <=0.01 KOR / <=0.03 ESP).
TASK_SPLIT_START_LEVEL = 1
TASK_SPLIT_BOUNDS_AREA = 2e11
TASK_SPLIT_AREA_NO_POP = 1e12
TASK_SPLIT_MAX_ADMINS = 100_000


def census_rake_main(
    resolution: str,
    version: str,
    iso3: str,
    census_time_point: str,
    task_parent_id: str,
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
    if verbose:
        logger.info("Starting")
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    if verbose:
        logger.info("Loading census inputs")
    task_admins, census_weights = pm_data.load_census_raking_inputs(
        model_spec,
        iso3=iso3,
        census_time_point=census_time_point,
        task_parent_id=task_parent_id,
    )
    task_map = task_admins.loc[[iso3], [census_time_point], [task_parent_id], :].reset_index().T[0].rename("task_arg")
    census_weights = census_weights.loc[iso3, :, census_time_point]["weight"]
    model_time_points = sorted(census_weights.index.to_list())
    # The v3 mechanism tracks pixel run-lengths, so it needs a few time points
    # beyond the written window (warm-up before it, the anchor window after the
    # census time point); rasters are still written only for the weighted tps.
    margin_tps = utils.margin_time_points(
        model_time_points,
        census_time_point,
        pm_data.list_raw_prediction_time_points(resolution, version),
    )
    rf_prior = pm_data.load_census_rf_prior(iso3, census_time_point, model_spec)
    required = {"pooling_level", "density_bound"}
    if missing_cols := required - set(rf_prior.columns):
        msg = (
            f"rf_prior predates the current pre-stage schema (missing "
            f"{sorted(missing_cols)}); delete raked_census/rf_prior/ and rerun "
            "the census_rake orchestrator to regenerate the pre-stage."
        )
        raise ValueError(msg)
    rf_country = float(rf_prior["rf_country"].iloc[0])
    pooling_level = int(rf_prior["pooling_level"].iloc[0])
    # Per-parent pooled rates from the pre-stage masses (persons-form; see
    # utils.RF_LAMBDA_PERSONS). rf_country <= 0 means no predicted mass
    # anywhere: no rates exist and every unit is gated off regardless.
    lam = utils.RF_LAMBDA_PERSONS
    if rf_country > 0:
        rf_values = (rf_prior["census_population"] + lam) / (
            rf_prior["predicted_population"] + lam / rf_country
        )
    else:
        rf_values = rf_prior["census_population"] * 0.0
    rf_by_parent = pd.Series(
        rf_values.to_numpy(dtype=float), index=rf_prior["pooling_parent_id"].to_numpy()
    )
    # Per-parent density-cap bounds, computed whole-census by the pre-stage
    # (NaN = no well-measured units = uncapped; the lookup fills inf).
    bounds_by_parent = pd.Series(
        rf_prior["density_bound"].to_numpy(dtype=float),
        index=rf_prior["pooling_parent_id"].to_numpy(),
    )

    if verbose:
        logger.info(
            "Processing census and prediction data\n"
            f'    Number of admins: {task_map["most_detailed_units"]:,}\n'
            f'    Population: {int(task_map["population_total"]):,}\n'
            f'    Area: {int(task_map["area"] / 1e6):,} km^2\n'
            f'    Bounding Box Area: {int(task_map["bounds_area"] / 1e6):,} km^2'
        )
    overlay_skeleton, prediction_data, template_raster, shape_ids, pooling_ids = utils.process_census_data(
        resolution,
        version,
        pm_data,
        task_map,
        margin_tps,
        pooling_level,
    )

    if (
        prediction_data is not None
        and overlay_skeleton is not None
        and shape_ids is not None
        and pooling_ids is not None
    ):
        if verbose:
            logger.info("Raking")
        shape_table, raked_rasters = utils.rake(
            overlay_skeleton,
            prediction_data,
            template_raster,
            census_time_point,
            margin_tps,
            model_time_points,
            pooling_ids,
            rf_by_parent,
            bounds_by_parent,
            rf_country,
        )
        shape_table.insert(0, "shape_id", shape_table.index.map(shape_ids))
        if shape_table["shape_id"].isna().any():
            msg = "shape code -> shape_id mapping failed (order mismatch)"
            raise ValueError(msg)
        pm_data.save_raked_census_table(
            shape_table.reset_index(drop=True),
            iso3,
            task_parent_id,
            census_time_point,
            model_spec,
        )
        if verbose:
            logger.info("Saving raked rasters")
        for model_time_point, raked_raster in zip(model_time_points, raked_rasters):
            pm_data.save_raked_census(
                raked_raster,
                iso3,
                task_parent_id,
                model_time_point,
                census_time_point,
                model_spec,
            )
    else:
        # No population: save the template for every time point. For an admin with
        # no predicted pixels this is a minimal all-nodata raster (see
        # process_census_data); otherwise a zero raster over the valid pixels.
        if verbose:
            logger.info("Saving rasters (no population)")
        for model_time_point in model_time_points:
            pm_data.save_raked_census(
                template_raster,
                iso3,
                task_parent_id,
                model_time_point,
                census_time_point,
                model_spec,
            )

    if verbose:
        logger.info("Complete")


def _census_rf_block_sums(
    args: tuple[
        str, str, str, str, str, list[tuple[str, shapely.Geometry]],
        str, str, int, list[str], shapely.Geometry,
    ],
) -> tuple[dict[str, float], pd.DataFrame | None]:
    """One block's pre-stage pixel sums: parent masses and unit footprints.

    Two products from the same block grid:
      * per pooling parent, the summed raw prediction at the census time point
        (RF_p denominator). Parent geometries arrive pre-clipped to the block's
        bounds; partial sums are added across blocks by the caller.
      * per most-detailed unit, the count of ON pixels (prediction > 0) at each
        anchor-window quarter (the density-cap footprint). Units are bbox-read
        here in the worker -- the whole-country census is too large to load
        once (USA ~40 GB) -- and a unit straddling blocks gets partial counts
        summed by the caller. Each unit is rasterized ALONE over its own bbox
        window with all_touched=True: the footprint must reproduce the tasks'
        any-intersection n_on convention (every pixel the unit touches), which
        the bound is applied against. A shared one-pass partition undercounts
        boundary pixels and inflated bounds ~2.8x median on KOR (measured
        2026-09-15, check_prestage_bounds.py); windowed per-unit masks match
        the overlay counts while staying cheap (the JPN/PHL blow-up was
        FULL-GRID per-shape masks, not windowed ones).

    The parent masses are rasterized in ONE pass (integer codes + bincount).
    """
    (
        resolution, version, output_dir, block_key, census_time_point,
        parent_geoms, iso3, year, most_detailed_level, window_tps, block_box,
    ) = args
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    rasters = {
        tp: pm_data.load_raw_prediction(block_key, tp, model_spec)
        for tp in window_tps
    }
    raster = rasters[census_time_point]
    array = np.nan_to_num(raster.to_numpy(), nan=0.0)
    codes = features.rasterize(
        [(geometry, i + 1) for i, (_, geometry) in enumerate(parent_geoms)],
        out_shape=raster.shape,
        transform=raster.transform,
        dtype="uint32",
    )
    sums = np.bincount(
        codes.ravel(), weights=array.ravel(), minlength=len(parent_geoms) + 1
    )
    parent_sums = {
        task_parent_id: float(sums[i + 1])
        for i, (task_parent_id, _) in enumerate(parent_geoms)
    }

    units = pm_data.load_census_data(
        iso3, year, bounds=block_box, admin_level=most_detailed_level
    )
    if units.empty:
        return parent_sums, None
    units = units.to_crs(raster.crs)
    on_arrays = {
        tp: np.nan_to_num(rasters[tp].to_numpy(), nan=0.0) > 0 for tp in window_tps
    }
    n_rows, n_cols = raster.shape
    inverse = ~raster.transform
    counts = {tp: np.zeros(len(units), dtype=np.int64) for tp in window_tps}
    for i, geometry in enumerate(units["geometry"]):
        minx, miny, maxx, maxy = geometry.bounds
        c0f, r0f = inverse * (minx, maxy)
        c1f, r1f = inverse * (maxx, miny)
        r0, c0 = max(0, int(np.floor(r0f))), max(0, int(np.floor(c0f)))
        r1, c1 = min(n_rows, int(np.ceil(r1f))), min(n_cols, int(np.ceil(c1f)))
        if r0 >= r1 or c0 >= c1:
            continue
        window_transform = raster.transform * Affine.translation(c0, r0)
        mask = features.rasterize(
            [(geometry, 1)],
            out_shape=(r1 - r0, c1 - c0),
            transform=window_transform,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        for tp in window_tps:
            counts[tp][i] = np.count_nonzero(mask & on_arrays[tp][r0:r1, c0:c1])
    footprints: dict[str, Any] = {"shape_id": units["shape_id"].to_numpy()}
    for tp in window_tps:
        footprints[f"n_on_{tp}"] = counts[tp]
    return parent_sums, pd.DataFrame(footprints)


# Empirical-lambda probe: the pooling constant is "measured, not chosen", so
# every census_rf task re-derives it from its own parent rates, persists the
# estimate for review, and fails loudly when the reading is trustworthy AND far
# outside the validated setting. Calibrated against the 2026_08_06.001 priors:
# legitimate estimates span ~5 orders of magnitude across all censuses, but the
# constant only binds -- and the estimator only interpolates rather than
# extrapolates -- for county-like censuses (many parents, median parent small),
# where the majors read USA 194 / JPN 230 / BRA 669 / MEX 2,793 around the
# configured 800 and the widest legitimate outliers (AUS 27, CAN 30) sit ~30x
# low. Hence: hard-fail only for county-like censuses (>=300 parents, median
# parent <= 100k people) beyond a 50x band -- zero false alarms on the
# calibration set. Scope: a UNIFORM unit rescale is invisible to this probe by
# the same invariance that makes it harmless to the persons-form mechanism;
# what the guard catches is noise/heterogeneity regime drift, and the
# estimator attenuates such shifts (a synthetic 250x true shift reads ~25x),
# so the 50x band corresponds to substantially larger true regime changes.
# Everything else prints + persists only.
LAMBDA_PROBE_TOLERANCE = 50.0
LAMBDA_PROBE_MIN_PARENTS = 30
LAMBDA_PROBE_MIN_SPREAD = 10.0
# The estimate is bin-count sensitive (a single q can move it 10-60x on real
# censuses), so the probe takes the median over several bin counts.
LAMBDA_PROBE_BIN_COUNTS = (6, 8, 12)
LAMBDA_PROBE_RAISE_MIN_PARENTS = 300
LAMBDA_PROBE_RAISE_MAX_MEDIAN_PARENT = 100_000.0


def probe_empirical_lambda(
    rf_prior: pd.DataFrame, iso3: str, census_time_point: str
) -> dict[str, float | str]:
    """Estimate the persons-lambda the setting rule implies for this census.

    Under the noise form the shrinkage itself assumes (rate variance ~ phi /
    persons), squared deviations of parent rates from the country rate satisfy
    E[(R_p - RF_country)^2] = tau^2 + phi / C_p. A robust line fit (medians
    within parent-size bins -- the median's scale factor cancels in the ratio)
    gives lambda_empirical = phi / tau^2, the mass at which a parent's own-rate
    noise equals the true cross-parent dispersion. Returns the estimate and a
    status for persistence; raises only for county-like censuses (see the
    constants above) beyond LAMBDA_PROBE_TOLERANCE-fold of RF_LAMBDA_PERSONS.
    """
    tag = f"{iso3} {census_time_point}"
    d = rf_prior.loc[
        (rf_prior["census_population"] > 0) & (rf_prior["predicted_population"] > 0)
    ]
    persons = d["census_population"].to_numpy(dtype=float)
    if len(d) < LAMBDA_PROBE_MIN_PARENTS:
        print(f"lambda probe {tag} skipped: {len(d)} parents < {LAMBDA_PROBE_MIN_PARENTS}")
        return {"lambda_empirical": np.nan, "probe_status": "too_few_parents"}
    if persons.max() / persons.min() < LAMBDA_PROBE_MIN_SPREAD:
        print(f"lambda probe {tag} skipped: insufficient parent-size spread")
        return {"lambda_empirical": np.nan, "probe_status": "no_size_spread"}
    rates = d["census_population"].to_numpy(dtype=float) / d["predicted_population"].to_numpy(dtype=float)
    rf_country = d["census_population"].sum() / d["predicted_population"].sum()
    x_all = 1.0 / persons
    y_all = (rates - rf_country) ** 2
    estimates = []
    for q in LAMBDA_PROBE_BIN_COUNTS:
        bins = pd.qcut(persons, q=q, duplicates="drop")
        binned = pd.DataFrame({"x": x_all, "y": y_all, "size_bin": bins.codes}).groupby("size_bin").median()
        if len(binned) < 3:
            continue
        phi, tau_sq = np.polyfit(binned["x"].to_numpy(), binned["y"].to_numpy(), 1)
        # tau^2 <= 0: rates are homogeneous (pooling harder than the constant
        # is harmless); phi <= 0: deviations GROW with size (systematic, not
        # sampling noise -- the rule's noise model doesn't apply). Neither is
        # the regime break this guard exists for; drop this bin count's fit.
        if phi > 0 and tau_sq > 0:
            estimates.append(float(phi / tau_sq))
    if not estimates:
        print(
            f"lambda probe {tag} not identifiable at any bin count; "
            "using RF_LAMBDA_PERSONS"
        )
        return {"lambda_empirical": np.nan, "probe_status": "not_identifiable"}
    lambda_empirical = float(np.median(estimates))
    county_like = (
        len(d) >= LAMBDA_PROBE_RAISE_MIN_PARENTS
        and float(np.median(persons)) <= LAMBDA_PROBE_RAISE_MAX_MEDIAN_PARENT
    )
    print(
        f"lambda probe {tag}: empirical {lambda_empirical:,.0f} persons vs "
        f"configured {utils.RF_LAMBDA_PERSONS:,.0f} ({len(d):,} parents, "
        f"median {np.median(persons):,.0f} people, "
        f"{'county-like: guarded' if county_like else 'informational only'})"
    )
    ratio = lambda_empirical / utils.RF_LAMBDA_PERSONS
    if county_like and not (1 / LAMBDA_PROBE_TOLERANCE <= ratio <= LAMBDA_PROBE_TOLERANCE):
        msg = (
            f"Empirical pooling mass for {tag} ({lambda_empirical:,.0f} persons) "
            f"deviates more than {LAMBDA_PROBE_TOLERANCE:g}x from "
            f"RF_LAMBDA_PERSONS ({utils.RF_LAMBDA_PERSONS:,.0f}) in a "
            "county-like census where the constant binds: the model's noise/"
            "heterogeneity regime no longer matches the validated setting -- "
            "re-measure the pooling constant before trusting construction-"
            "channel credits."
        )
        raise ValueError(msg)
    return {
        "lambda_empirical": lambda_empirical,
        "probe_status": "guarded" if county_like else "informational",
    }


def census_rf_main(
    resolution: str,
    version: str,
    iso3: str,
    census_time_point: str,
    output_dir: str | Path,
    num_cores: int,
) -> None:
    """RF pooling prior for one census: the country rate the v3 mechanism's
    parent rates pool toward, RF_country = census total / predicted total at
    the census time point, plus per-parent predicted masses as diagnostics."""
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    # Semantic pooling parents: one admin level above the census's most
    # detailed, capped at admin2 (national -- the level-0 row -- when admin1 is
    # the most detailed). One "parent" everywhere: RF pooling groups and the
    # density-cap families both use this level. Census masses are sums of the
    # most-detailed units (the units actually raked, matching the in-task sums
    # the mechanism uses) rather than the parent rows' own population_total,
    # which can be NaN or disagree with the children in some hierarchies.
    year = census_time_point.split("q")[0]
    hierarchy = pd.read_parquet(
        pm_data.census_path(iso3, year),
        columns=["shape_id", "admin_level", "path_to_top_parent", "population_total"],
    )
    most_detailed_level = int(hierarchy["admin_level"].max())
    pooling_level = max(0, min(2, most_detailed_level - 1))
    children = hierarchy.loc[
        hierarchy["admin_level"] == most_detailed_level,
        ["shape_id", "path_to_top_parent", "population_total"],
    ].set_index("shape_id")
    children["parent"] = children["path_to_top_parent"].str.split(",").str[pooling_level]
    census_by_parent = children.groupby("parent")["population_total"].sum()
    parents = pm_data.load_census_data(iso3, year, admin_level=pooling_level).loc[
        :, ["shape_id", "geometry"]
    ].rename(columns={"shape_id": "pooling_parent_id"})
    if parents["pooling_parent_id"].duplicated().any():
        # A duplicate would silently merge-and-drop predicted mass in the
        # per-block dict accumulation, then crash the tasks' Series.map far
        # from the cause.
        dupes = parents.loc[parents["pooling_parent_id"].duplicated(), "pooling_parent_id"]
        msg = f"duplicate pooling parent ids in census file: {sorted(set(dupes))[:5]}"
        raise ValueError(msg)
    print(
        f"{iso3} {census_time_point}: most detailed level {most_detailed_level} "
        f"-> pooling level {pooling_level}, {len(parents):,} parents"
    )

    # Anchor-window quarters (same helper the tasks use, so the footprint is
    # measured over exactly the quarters that define existing stock).
    available_tps = sorted(pm_data.list_raw_prediction_time_points(resolution, version))
    window_tps = utils.anchor_window(census_time_point, available_tps)

    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_bounds = (
        modeling_frame.bounds.assign(block_key=modeling_frame["block_key"].to_numpy())
        .groupby("block_key")
        .agg({"minx": "min", "miny": "min", "maxx": "max", "maxy": "max"})
    )
    sindex = parents.sindex
    block_args = []
    for block_key, b in block_bounds.iterrows():
        block_box = shapely.box(b.minx, b.miny, b.maxx, b.maxy)
        hit = parents.iloc[sindex.query(block_box, predicate="intersects")]
        if hit.empty:
            continue
        # Keep only polygonal clip results: a parent touching the block along a
        # shared edge clips to a line/point, which rasterize would burn as a
        # spurious pixel row (order-dependently stealing another parent's mass).
        parent_geoms = []
        for task_parent_id, geometry in zip(
            hit["pooling_parent_id"], hit["geometry"], strict=True
        ):
            clipped = geometry.intersection(block_box)
            if clipped.geom_type == "GeometryCollection":
                polys = [g for g in clipped.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                clipped = shapely.union_all(polys) if polys else clipped
            if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            parent_geoms.append((task_parent_id, clipped))
        if not parent_geoms:
            continue
        block_args.append(
            (
                resolution, version, str(output_dir), str(block_key),
                census_time_point, parent_geoms,
                iso3, year, most_detailed_level, window_tps, block_box,
            )
        )

    print(f"Summing predictions over {len(block_args)} blocks at {census_time_point}")
    block_sums = parallel.run_parallel(
        _census_rf_block_sums,
        block_args,
        num_cores=num_cores,
    )
    predicted: dict[str, float] = {}
    unit_frames = []
    for parent_sums, units_df in block_sums:
        for task_parent_id, value in parent_sums.items():
            predicted[task_parent_id] = predicted.get(task_parent_id, 0.0) + value
        if units_df is not None:
            unit_frames.append(units_df)

    # Growth-density cap bounds per pooling parent, over the WHOLE census (no
    # task can truncate a family): unit footprint = max across the anchor-window
    # quarters of its ON-pixel count (partial counts summed across blocks
    # first); family density = census / footprint over well-measured units;
    # bound = q99 for families of >= DENSITY_CAP_MIN_UNITS, else k x family max
    # (constants + provenance in utils). Parents with no well-measured units
    # stay NaN = uncapped.
    if not unit_frames:
        msg = (
            f"{iso3} {census_time_point}: no most-detailed census units "
            "intersect any prediction block; cannot compute density bounds."
        )
        raise ValueError(msg)
    per_tp = pd.concat(unit_frames, ignore_index=True).groupby("shape_id").sum()
    footprint = per_tp.max(axis=1).reindex(children.index).fillna(0.0)
    well_measured = (
        children["population_total"] >= utils.DENSITY_CAP_MIN_CENSUS
    ) & (footprint > 0)
    densities = (
        children.loc[well_measured, "population_total"]
        / footprint[well_measured]
    )

    def _family_bound(dens: "pd.Series[Any]") -> float:
        if len(dens) >= utils.DENSITY_CAP_MIN_UNITS:
            return float(np.quantile(dens, utils.DENSITY_CAP_QUANTILE))
        return float(dens.max()) * utils.DENSITY_CAP_SMALL_FAMILY_MULT

    bounds_by_parent = densities.groupby(
        children.loc[well_measured, "parent"]
    ).apply(_family_bound)
    print(
        f"{iso3} {census_time_point}: density bounds for "
        f"{len(bounds_by_parent):,} of {len(census_by_parent):,} parents "
        f"({int(well_measured.sum()):,} well-measured units)"
    )

    rf_prior = parents.loc[:, ["pooling_parent_id"]].copy()
    rf_prior["census_population"] = (
        rf_prior["pooling_parent_id"].map(census_by_parent).fillna(0.0)
    )
    rf_prior["predicted_population"] = (
        rf_prior["pooling_parent_id"].map(predicted).fillna(0.0)
    )
    rf_prior["pooling_level"] = pooling_level
    rf_prior["density_bound"] = rf_prior["pooling_parent_id"].map(bounds_by_parent)
    census_total = rf_prior["census_population"].sum()
    predicted_total = rf_prior["predicted_population"].sum()
    rf_prior["rf_country"] = (
        census_total / predicted_total if predicted_total > 0 else 0.0
    )
    print(
        f"{iso3} {census_time_point}: census {census_total:,.0f} / predicted "
        f"{predicted_total:,.1f} -> rf_country {rf_prior['rf_country'].iloc[0]:.4f}"
    )
    # Shared-trigger surface (print-only; the diagnostics stage carries the
    # FLAG): a parent whose predicted mass is prior-dominated (M_p < L/rf_c,
    # i.e. the pooled rate leans more on the prior than on data -- a derived
    # threshold, not a tuned one) AND whose density bound is NaN has BOTH
    # backstops off at once: RF_p ~ rf_c*(C_p+L)/L is unbounded in C_p and no
    # cap constrains later credit. The incumbent's near-zero-denominator
    # pathology can reappear at parent scale exactly here.
    rf_c = float(rf_prior["rf_country"].iloc[0])
    if rf_c > 0:
        prior_dominated = (
            rf_prior["predicted_population"] < utils.RF_LAMBDA_PERSONS / rf_c
        )
        shared_trigger = (
            prior_dominated
            & rf_prior["density_bound"].isna()
            & (rf_prior["census_population"] > 0)
        )
        if shared_trigger.any():
            print(
                f"WARNING {iso3} {census_time_point}: "
                f"{int(shared_trigger.sum()):,} parents are prior-dominated AND "
                f"uncapped ({rf_prior.loc[shared_trigger, 'census_population'].sum():,.0f} "
                "census people) -- construction credit there is bounded by "
                "neither pooling evidence nor a density cap."
            )

    # Probe BEFORE saving: a failed probe must not leave a prior on disk, or
    # the orchestrator's skip-if-exists would mask the failure on relaunch.
    probe = probe_empirical_lambda(rf_prior, iso3, census_time_point)
    rf_prior["lambda_empirical"] = probe["lambda_empirical"]
    rf_prior["lambda_probe_status"] = probe["probe_status"]
    pm_data.save_census_rf_prior(rf_prior, iso3, census_time_point, model_spec)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_iso3()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
def census_rf_task(
    resolution: str,
    version: str,
    iso3: str,
    time_point: str,
    output_dir: str | Path,
    num_cores: int,
) -> None:
    census_rf_main(
        resolution=resolution,
        version=version,
        iso3=iso3,
        census_time_point=time_point,
        output_dir=output_dir,
        num_cores=num_cores,
    )


def check_time_points(census_weights: pd.DataFrame) -> None:
    test = census_weights.reset_index()
    test["match"] = test["model_time_point"] == test["census_time_point"]
    test = test.groupby(["iso3", "census_time_point"])["match"].any()

    if not test.all():
        raise ValueError(f"Missing raw_predictions for at least one time-point found in census data:\n{test}")


def check_complete(
    census_task: tuple[str, str, str],
    pm_data: PopulationModelData,
    model_spec: ModelSpecification,
):
    iso3, census_time_point, task_parent_id = census_task
    # A completed task always writes a raster at model_time_point == census_time_point
    # (weight 1 there). Use that as the marker: with multiple censuses per country a
    # census contributes to only a subset of model time points, so a fixed global last
    # time point would be missing for early censuses and they'd rerun forever.
    path = pm_data.raked_census_path(iso3, census_time_point, census_time_point, model_spec) / f"{task_parent_id}.tif"
    if not path.exists():
        return ()
    elif path.stat().st_size < 1:
        return ()
    else:
        return iso3, census_time_point, task_parent_id


# Per-task resources are predicted from each task's admin features, replacing the
# old xxl/xl/big/standard tiers (one workflow, individually sized tasks). Model
# calibrated on the 2026-07-06 full-run jobmon metadata (17,073 done tasks; see
# .claude/validate_census_rake/calibrate_per_task_resources.py):
#   MEMORY = margin * (floor
#            + census-cache term: the country census parquet is scanned through
#              the page cache each task (USA 11.7 GB -> ~26 GB floor)
#            + box term ~ bounds_area: template + scatter arrays span the full
#              bounding box (pop-0 maritime/arctic admins: huge bbox, little else)
#            + covered-land term ~ area for POPULATED admins: the covered_pixels
#              x n_tps working set (~0.22-0.24 GB per 1000 km^2, stable from 50k
#              to 522k km^2); population==0 admins are heavily water-masked, so
#              a small slope. NOTE a future *empty* fully-covered desert would be
#              under-predicted and lean on the retry.)
#   RUNTIME = margin * (floor + area term + perimeter term (convoluted CAN
#             boundaries: high runtime at low memory)).
# Coverage on the full run: 97.6% of tasks fit the first attempt; the 2.4% tail
# (worst obs/request ratio 1.24) is fully covered by jobmon's default +50% retry
# bump -- margins-not-maxima beats zero-retry tiers: 75% of the tiers' reserved
# memory and 52% of their reserved runtime. SAU.8_1 (the 96 G kill, est ~115 GB)
# gets a 166 G first attempt.
MEMORY_FLOOR_GB = 3.5
MEMORY_PER_CENSUS_GB = 2.0
MEMORY_PER_KKM2_BOX = 0.012
MEMORY_PER_KKM2_POPULATED = 0.24
MEMORY_PER_KKM2_EMPTY = 0.04
MEMORY_MARGIN = 1.15
MEMORY_BOUNDS_GB = (8, 240)
RUNTIME_FLOOR_MIN = 6.0
RUNTIME_PER_KKM2 = 0.09
RUNTIME_PER_KKM_PERIM = 2.5
RUNTIME_MARGIN = 1.7
RUNTIME_BOUNDS_MIN = (10, 240)


def build_task_resources(
    pm_data: PopulationModelData,
    task_admins: gpd.GeoDataFrame,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Predicted memory/runtime per (iso3, census_time_point, task_parent_id)."""
    features = task_admins.reset_index()
    census_gb = {
        (iso3, ctp): pm_data.census_path(iso3, ctp.split("q")[0]).stat().st_size / 1e9
        for iso3, ctp in features[["iso3", "census_time_point"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    cache_gb = pd.Series(
        [census_gb[key] for key in zip(features["iso3"], features["census_time_point"], strict=True)],
        index=features.index,
    )
    area_kkm2 = features["area"] / 1e9
    bounds_kkm2 = features["bounds_area"] / 1e9
    perim_kkm = features["perimeter"] / 1e6
    populated = features["population_total"] > 0

    memory_gb = MEMORY_MARGIN * (
        MEMORY_FLOOR_GB
        + MEMORY_PER_CENSUS_GB * cache_gb
        + MEMORY_PER_KKM2_BOX * bounds_kkm2
        + np.where(populated, MEMORY_PER_KKM2_POPULATED, MEMORY_PER_KKM2_EMPTY) * area_kkm2
    )
    memory_gb = np.ceil(memory_gb.clip(*MEMORY_BOUNDS_GB)).astype(int)
    runtime_min = RUNTIME_MARGIN * (
        RUNTIME_FLOOR_MIN
        + RUNTIME_PER_KKM2 * area_kkm2
        + RUNTIME_PER_KKM_PERIM * perim_kkm
    )
    runtime_min = np.ceil(runtime_min.clip(*RUNTIME_BOUNDS_MIN)).astype(int)

    return {
        (iso3, ctp, tpid): {"memory": f"{mem}G", "runtime": f"{run}m"}
        for iso3, ctp, tpid, mem, run in zip(
            features["iso3"],
            features["census_time_point"],
            features["task_parent_id"],
            memory_gb,
            runtime_min,
            strict=True,
        )
    }


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_iso3()
@click.option("--task-parent-id", type=str, required=True)
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def census_rake_task(
    resolution: str,
    version: str,
    iso3: str,
    time_point: str,
    task_parent_id: str,
    output_dir: str | Path,
) -> None:
    census_rake_main(
        resolution=resolution,
        version=version,
        iso3=iso3,
        census_time_point=time_point,
        task_parent_id=task_parent_id,
        output_dir=output_dir,
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def census_rake(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = sorted(clio.convert_choice(time_point, prediction_time_points))

    # Thresholds are module constants with a dual compute/statistical role --
    # see TASK_SPLIT_* above before retuning.
    task_admins, census_weights = utils.generate_census_inputs(
        pm_data,
        start_level=TASK_SPLIT_START_LEVEL,
        bounds_area_threshold=TASK_SPLIT_BOUNDS_AREA,
        area_no_pop_threshold=TASK_SPLIT_AREA_NO_POP,
        admins_threshold=TASK_SPLIT_MAX_ADMINS,
    )
    census_weights = census_weights.loc[:, time_points, :]
    check_time_points(census_weights)

    census_tasks = (
        task_admins.reset_index()
        .loc[:, ["iso3", "census_time_point", "task_parent_id"]]
        .drop_duplicates()
    )

    pm_data.save_census_raking_metadata(
        task_admins,
        census_weights,
        census_tasks,
        model_spec
    )
    possible_census_tasks = list(census_tasks.itertuples(index=False, name=None))

    # Pre-stage: one small task per census computes the RF pooling prior
    # (RF_country = census total / predicted total at the census time point)
    # that every census_rake task pools its parent rate toward. Skipped where
    # the prior parquet already exists -- which makes "new raw predictions =>
    # new model version" load-bearing: regenerating predictions in place under
    # the same version would leave stale priors (and a disarmed lambda probe).
    rf_tasks = [
        (task_iso3, task_ctp)
        for task_iso3, task_ctp in census_tasks.loc[:, ["iso3", "census_time_point"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
        if not pm_data.census_rf_prior_path(task_iso3, task_ctp, model_spec).exists()
    ]
    if len(rf_tasks) > 0:
        print(f"Computing RF pooling priors for {len(rf_tasks):,} censuses.")
        jobmon.run_parallel(
            runner="pmtask postprocess",
            task_name="census_rf",
            task_resources={
                "queue": queue,
                "cores": 8,
                "memory": "16G",
                "runtime": "60m",
                "project": "proj_rapidresponse",
            },
            flat_node_args=(("iso3", "time-point"), rf_tasks),
            task_args={
                "resolution": resolution,
                "version": version,
                "output-dir": output_dir,
                "num-cores": 8,
            },
            max_attempts=2,
            log_root=pm_data.log_dir("postprocess_census_rf"),
        )

    _check_complete = functools.partial(
        check_complete,
        pm_data=pm_data,
        model_spec=model_spec,
    )
    complete_census_tasks = parallel.run_parallel(
        _check_complete,
        possible_census_tasks,
        num_cores=10,
    )
    complete_census_tasks = [i for i in complete_census_tasks if len(i) > 0]

    task_resources_by_task = build_task_resources(pm_data, task_admins)

    complete = set(complete_census_tasks)
    census_tasks = [task for task in possible_census_tasks if task not in complete]
    if 0 < DOWNSAMPLE_ADMINS < len(census_tasks):
        census_tasks = pd.Series(census_tasks).sample(DOWNSAMPLE_ADMINS).sort_index().tolist()

    if len(census_tasks) > 0:
        print(
            f"Building raking factors for {len(census_tasks):,} census time-admins "
            f"(out of a possible {len(possible_census_tasks):,})."
        )
        jobmon.run_parallel(
            runner="pmtask postprocess",
            task_name="census_rake",
            task_resources={
                "queue": queue,
                "cores": 1,
                "memory": f"{MEMORY_BOUNDS_GB[0]}G",
                "runtime": f"{RUNTIME_BOUNDS_MIN[0]}m",
                "project": "proj_rapidresponse",
            },
            flat_node_args=(("iso3", "time-point", "task-parent-id"), census_tasks),
            per_task_resources=lambda args: task_resources_by_task[tuple(args)],
            task_args={
                "resolution": resolution,
                "version": version,
                "output-dir": output_dir,
            },
            max_attempts=2,
            log_root=pm_data.log_dir("postprocess_census_rake"),
            concurrency_limit=1_000,
        )
