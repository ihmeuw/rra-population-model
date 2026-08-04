from collections import defaultdict
from typing import Any, NamedTuple

import click
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import tqdm
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon, parallel

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import (
    block_census_tasks,
    get_prediction_time_point,
    load_block_census_layer,
    paste_on_canvas,
)

RAKING_VERSION = "gbd_2023"
STAGES = ["initial", "final"]
# For a GBD admin touched by no census, the spliced field is exactly rf1 * raw,
# so its final factor must be ~1 (true / (rf1 * raw) with the same sums that
# produced rf1). A violation means the two stages disagree on the field being
# summed -- i.e. a pipeline bug -- so it raises rather than warns.
NO_CENSUS_RF_TOLERANCE = 1e-3


def load_admin_populations(
    pm_data: PopulationModelData,
    time_point: str,
) -> gpd.GeoDataFrame:
    raking_pop = pm_data.load_raking_population(version=RAKING_VERSION)
    all_pop = raking_pop.loc[raking_pop.most_detailed == 1].set_index(
        ["year_id", "location_id"]
    )["population"]

    # Interpolate the time point population
    if "q" in time_point:
        year, quarter = (int(s) for s in time_point.split("q"))

        if RAKING_VERSION == "gbd_2023":
            max_data_year = all_pop.index.get_level_values("year_id").max()
            next_year = year + 1
            if next_year > 2027:
                raise ValueError("Don't project beyond 2026")
            if next_year > max_data_year:
                prior_year_pop = all_pop.loc[max_data_year - (year - max_data_year)]
                next_year_pop = all_pop.loc[max_data_year - (next_year - max_data_year)]
                max_data_year_pop = all_pop.loc[max_data_year]
                prior_year_pop = max_data_year_pop * (max_data_year_pop / prior_year_pop)
                next_year_pop = max_data_year_pop * (max_data_year_pop / next_year_pop)
            else:
                prior_year_pop = all_pop.loc[year]
                next_year_pop = all_pop.loc[next_year]
        else:
            next_year = min(year + 1, 2100)
            prior_year_pop = all_pop.loc[year]
            next_year_pop = all_pop.loc[next_year]

        weight = (int(quarter) - 1) / 4

        pop = ((1 - weight) * prior_year_pop + weight * next_year_pop).reset_index()
    else:
        year = int(time_point)
        pop = all_pop.loc[year]

    admins = pm_data.load_raking_shapes(version=RAKING_VERSION)
    pop = admins[["location_id", "geometry"]].merge(pop, on="location_id")
    return pop


class AggregationArgs(NamedTuple):
    resolution: str
    model_version: str
    block_key: str
    time_point: str
    output_dir: str
    shape_map: dict[int, shapely.Polygon]
    tile_poly: shapely.Polygon
    # None -> initial stage (no census layer); a (possibly empty) list -> final
    # stage, holding the (iso3, task_parent_id, census_time_point) admins whose
    # rasters touch this block. Identity tuples, not geometries, so the pickled
    # payload per worker stays small.
    census_tasks: list[tuple[str, str, str]] | None
    census_weights: pd.DataFrame | None


def aggregate_unraked_population(
    aggregate_args: AggregationArgs,
) -> dict[int, tuple[float, float, float]]:
    """Per-admin sums over one block: (raw, census, raw-under-census).

    The initial stage only uses the raw sum. The final stage combines all three
    into the spliced-field sum: census + rf1 * (raw - raw_under_census) -- the
    census layer wins wherever it has valid data (matching the rake stage's
    ``merge first``), and rf1 is constant within an admin, so the multiplied
    raster never needs to be materialized.
    """
    est_pop: dict[int, tuple[float, float, float]] = {}
    (
        resolution,
        model_version,
        block_key,
        time_point,
        output_dir,
        shape_map,
        tile_poly,
        census_tasks,
        census_weights,
    ) = aggregate_args
    if not shape_map:
        return est_pop

    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, model_version)
    r = pm_data.load_raw_prediction(block_key, time_point, model_spec)

    cens_arr = cens_valid = None
    if census_tasks and census_weights is not None:
        census_layer = load_block_census_layer(
            pm_data,
            model_spec,
            tile_poly,
            time_point,
            census_tasks,
            census_weights,
        )
        if census_layer is not None:
            cens_arr = paste_on_canvas(census_layer, r).to_numpy()
            cens_valid = ~np.isnan(cens_arr)

    raw_arr = r.to_numpy()
    for location_id, geom in shape_map.items():
        shape_mask, *_ = raster_geometry_mask(
            data_transform=r.transform,
            data_width=r.shape[1],
            data_height=r.shape[0],
            shapes=[geom],
            invert=True,
        )
        raw_sum = float(np.nansum(raw_arr[shape_mask]))
        census_sum = raw_cens_sum = 0.0
        if cens_arr is not None and cens_valid is not None:
            covered = shape_mask & cens_valid
            census_sum = float(np.nansum(cens_arr[covered]))
            raw_cens_sum = float(np.nansum(raw_arr[covered]))
        est_pop[location_id] = (raw_sum, census_sum, raw_cens_sum)
    return est_pop


def raking_factors_main(
    resolution: str,
    version: str,
    time_point: str,
    stage: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("loading model frame")
    model_spec = pm_data.load_model_specification(resolution, version)
    model_frame = pm_data.load_modeling_frame(resolution)
    population = load_admin_populations(pm_data, time_point)
    population = population.to_crs(model_frame.crs)
    prediction_time_point = get_prediction_time_point(
        pm_data, resolution, version, time_point
    )

    rf_initial: "pd.Series[Any]" | None = None
    task_admins = census_weights = None
    if stage == "final":
        print("Loading initial raking factors and census metadata")
        rf_initial = (
            pm_data.load_raking_factors(time_point, model_spec, stage="initial")
            .drop_duplicates("location_id")
            .set_index("location_id")["raking_factor"]
        )
        task_admins, census_weights = pm_data.load_census_raking_inputs(model_spec)

    print("Building location aggregation args")
    block_keys = model_frame.block_key.unique().tolist()
    compute_location_args = []
    for block_key in tqdm.tqdm(block_keys, disable=not progress_bar):
        tile_poly = shapely.box(
            *model_frame[model_frame.block_key == block_key].total_bounds
        )
        shape_map = (
            population[population.intersects(tile_poly)]
            .clip(tile_poly)
            .set_index("location_id")
            .geometry.to_dict()
        )
        compute_location_args.append(
            AggregationArgs(
                resolution,
                version,
                block_key,
                prediction_time_point,
                str(output_dir),
                shape_map,
                tile_poly,
                block_census_tasks(task_admins, tile_poly) if stage == "final" else None,
                census_weights,
            )
        )

    print("Aggregating population")
    aggregate_pops_by_block = parallel.run_parallel(
        aggregate_unraked_population,
        compute_location_args,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    print("Collating aggregate population across blocks")
    aggregate_pops: dict[int, npt.NDArray[np.float64]] = defaultdict(lambda: np.zeros(3))
    for pop_dict in aggregate_pops_by_block:
        for location_id, sums in pop_dict.items():
            aggregate_pops[location_id] += np.asarray(sums)

    print("Calculating raking factors")
    combined = pd.DataFrame.from_dict(
        aggregate_pops, orient="index", columns=["raw", "census", "raw_census"]
    ).rename_axis("location_id")
    loc_pop = population.set_index("location_id").population
    combined["true"] = loc_pop.reindex(combined.index)
    if stage == "initial":
        combined["unraked"] = combined["raw"]
    else:
        rf1 = rf_initial.reindex(combined.index)  # type: ignore[union-attr]
        missing = combined.index.difference(
            rf_initial.index  # type: ignore[union-attr]
        ).tolist()
        if missing:
            raise ValueError(
                f"No initial raking factor for location_ids {missing}; the initial "
                "and final stages must be built from the same raking shapes."
            )
        # A present-but-NaN initial factor is legitimate, not a shape mismatch:
        # unmodeled supplement locations have NaN target populations and
        # zero-population locations are 0/0. NaN propagates through the spliced
        # field to a NaN final factor, which rake renders as the same unmodeled
        # carve-out (NaN pixels) the initial stage produces.
        # The spliced field: census wherever the census layer has valid data,
        # rf1 * raw elsewhere.
        combined["unraked"] = combined["census"] + rf1 * (
            combined["raw"] - combined["raw_census"]
        )
    combined["raking_factor"] = combined["true"] / combined["unraked"]

    if stage == "final":
        no_census = (combined["census"] == 0) & (combined["raw_census"] == 0)
        checkable = (
            no_census
            & np.isfinite(rf1)
            & (combined["raw"] > 0)
            & (combined["true"] > 0)
        )
        deviation = (combined.loc[checkable, "raking_factor"] - 1).abs()
        max_deviation = float(deviation.max()) if checkable.any() else 0.0
        print(
            f"census-touched admins: {int((~no_census).sum()):,} of {len(combined):,}; "
            f"max |rf - 1| over untouched admins: {max_deviation:.2e}"
        )
        if max_deviation > NO_CENSUS_RF_TOLERANCE:
            worst = deviation.idxmax()
            raise ValueError(
                "Final raking factor deviates from 1 for admins untouched by any "
                f"census (worst: location_id {worst}, |rf - 1| = {max_deviation:.3g}). "
                "The initial and final stages disagree on the unspliced field -- "
                "this indicates a pipeline bug, not a data property."
            )

    print("Building raking args")
    unraked_pop = combined["unraked"].to_dict()
    true_pop = combined["true"].to_dict()
    raking_factor = combined["raking_factor"].to_dict()
    raking_factors_by_block = []
    for loc_args in tqdm.tqdm(compute_location_args, disable=not progress_bar):
        for location_id, geom in loc_args.shape_map.items():
            raking_factors_by_block.append(
                (
                    loc_args.block_key,
                    location_id,
                    unraked_pop[location_id],
                    true_pop[location_id],
                    raking_factor[location_id],
                    geom,
                )
            )

    print("Collating")
    final_raking_factors = gpd.GeoDataFrame(
        raking_factors_by_block,
        columns=[
            "block_key",
            "location_id",
            "raw_pop",
            "true_pop",
            "raking_factor",
            "geometry",
        ],
        crs=model_frame.crs,
    )
    print("Saving")
    pm_data.save_raking_factors(final_raking_factors, time_point, model_spec, stage)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None)
@click.option("--stage", type=click.Choice(STAGES), required=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_progress_bar()
def raking_factors_task(
    resolution: str,
    version: str,
    time_point: str,
    stage: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,
) -> None:
    raking_factors_main(
        resolution, version, time_point, stage, output_dir, num_cores, progress_bar
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_copy_from_version()
@clio.with_time_point(choices=None, allow_all=True)
@click.option("--extrapolate", is_flag=True)
@click.option("--stage", type=click.Choice(STAGES), required=True)
@click.option(
    "--q1-only",
    is_flag=True,
    help="Subset to q1 time points. Validation-only versions: any version that "
    "produces final outputs needs initial factors at every time point.",
)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_num_cores(default=8)
@clio.with_queue()
def raking_factors(
    resolution: str,
    version: str,
    copy_from_version: str | None,
    time_point: str,
    extrapolate: bool,
    stage: str,
    q1_only: bool,
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    if q1_only and stage == "final":
        raise ValueError(
            "--q1-only is for validation-only versions; the final stage exists "
            "solely to serve the (complete) final rake."
        )
    pm_data = PopulationModelData(output_dir)
    pm_data.maybe_copy_version(resolution, version, copy_from_version)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = clio.convert_choice(time_point, prediction_time_points)
    if extrapolate:
        full_time_series = [f"{y}q1" for y in range(1950, 2101)]
        time_points = sorted(set(time_points) | set(full_time_series))
    if q1_only:
        time_points = [tp for tp in time_points if tp.endswith("q1")]

    if stage == "final":
        # Fail fast on missing inputs rather than mid-workflow: the final stage
        # needs the initial factors for every requested time point and the
        # census-raking metadata (written by the census_rake orchestrator).
        initial_time_points = set(
            pm_data.list_raking_factor_time_points(resolution, version, stage="initial")
        )
        missing = sorted(set(time_points) - initial_time_points)
        if missing:
            raise ValueError(
                f"No initial raking factors for time points {missing}; run "
                "`raking_factors --stage initial` first."
            )
        model_spec = pm_data.load_model_specification(resolution, version)
        pm_data.load_census_raking_inputs(model_spec)

    if resolution == "40":
        # The final stage additionally builds the block census layer and two
        # extra masked sums per admin on census-touched blocks.
        memory_per_core = 5 if stage == "initial" else 6
        runtime = "60m" if stage == "initial" else "90m"
        task_resources = {
            "queue": queue,
            "cores": num_cores,
            "memory": f"{num_cores * memory_per_core}G",
            "runtime": runtime,
            "project": "proj_rapidresponse",
        }
    elif resolution == "100":
        task_resources = {
            "queue": queue,
            "cores": num_cores,
            "memory": f"{int(num_cores * 2.5)}G",
            "runtime": "20m" if stage == "initial" else "40m",
            "project": "proj_rapidresponse",
        }

    print(f"Building {stage} raking factors for {len(time_points)} time points.")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="raking_factors",
        task_resources=task_resources,
        node_args={
            "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "num-cores": num_cores,
            "stage": stage,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_raking_factors"),
    )
