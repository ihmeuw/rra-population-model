import click
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
import shapely
from affine import Affine
from rasterio.features import MergeAlg, rasterize
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

# Coverage-weighted attribution: each pixel is split into SUPERSAMPLE^2 subcells
# and its population divided uniformly among them, so a boundary pixel is shared
# between shapes in proportion to (quantized) covered area -- the same
# attribution the census raking uses -- instead of winner-takes-all at the pixel
# center, which smears fine admin units (at 40m, most pixels of a US block or
# UK output area ARE boundary pixels). 4x -> 1/16-pixel area quantization, well
# below the irreducible noise floor from pixel-value blending.
SUPERSAMPLE = 4
# Coarse rows per bincount band: bounds the transient float64 weights copy so
# peak memory stays ~ the supersampled uint32 mask.
CHUNK_ROWS = 256


def coverage_weighted_sums(
    pop_raster: rt.RasterArray,
    shape_values: list[tuple[shapely.Polygon | shapely.MultiPolygon, int]],
    n_shapes: int,
) -> npt.NDArray[np.float64]:
    """Per-shape population sums with partial-pixel (area) attribution."""
    s = SUPERSAMPLE
    height, width = pop_raster.shape
    mask = rasterize(
        shape_values,
        out_shape=(height * s, width * s),
        transform=pop_raster.transform * Affine.scale(1 / s),
        merge_alg=MergeAlg.replace,
        dtype=np.uint32,
    )
    pop_arr = pop_raster._ndarray  # noqa: SLF001
    sums = np.zeros(n_shapes + 1)
    for r0 in range(0, height, CHUNK_ROWS):
        r1 = min(height, r0 + CHUNK_ROWS)
        weights = (
            np.repeat(
                np.repeat(np.nan_to_num(pop_arr[r0:r1], nan=0.0), s, axis=0), s, axis=1
            )
            / (s * s)
        )
        sums += np.bincount(
            mask[r0 * s : r1 * s].ravel(),
            weights=weights.ravel(),
            minlength=n_shapes + 1,
        )
    return sums[1:]


def build_bounds_map(
    raster_template: rt.RasterArray,
    shape_values: list[tuple[shapely.Polygon | shapely.MultiPolygon, int]],
) -> dict[int, tuple[slice, slice]]:
    # Still used by validate.comparison (center-point attribution); the metrics
    # stage itself uses coverage_weighted_sums above.
    # The tranform maps pixel coordinates to the CRS coordinates.
    # This mask is the inverse of that transform.
    to_pixel = ~raster_template.transform

    bounds_map = {}
    for shp, loc_id in shape_values:
        xmin, ymin, xmax, ymax = shp.bounds
        pxmin, pymin = to_pixel * (xmin, ymax)
        pixel_buffer = 10
        pxmin = max(0, int(pxmin) - pixel_buffer)
        pymin = max(0, int(pymin) - pixel_buffer)
        pxmax, pymax = to_pixel * (xmax, ymin)
        pxmax = min(raster_template.width, int(pxmax) + pixel_buffer)
        pymax = min(raster_template.height, int(pymax) + pixel_buffer)
        bounds_map[loc_id] = (slice(pymin, pymax), slice(pxmin, pxmax))

    return bounds_map


def pixel_metrics_main(
    block_key: str,
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Loading validation frame")
    validation_frame = pm_data.load_validation_frame(resolution)
    block_validation_frame = validation_frame.loc[
        validation_frame.block_key == block_key
    ]
    validation_locs = list(
        block_validation_frame[["iso3", "year"]].itertuples(index=False)
    )

    print("Loading and subsetting modeling frame")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    block_frame = (
        modeling_frame.loc[
            modeling_frame.block_key == block_key, ["block_key", "geometry"]
        ]
        .dissolve("block_key")
        .reset_index()
    )
    block_poly = block_frame.geometry.iloc[0]

    print("Loading GBD-raked population predictions")
    # The pseudo-OOS compares GBD-raked predictions against censuses, so it must
    # read the gbd_raked product (raw x initial raking factor, NO census splice)
    # -- the final raked_predictions have census information baked in, which
    # would make the comparison circular. Materialize the needed time points
    # with `rake --stage gbd` first.
    model_spec = pm_data.load_model_specification(resolution, version)
    pop_raster = pm_data.load_gbd_raked_prediction(block_key, time_point, model_spec)

    print("Loading and subsetting census data")
    iter_data = []
    for iso3, year in validation_locs:
        path = pm_data.census_path(iso3, year)
        max_admin_level = int(
            pd.read_parquet(path, columns=["admin_level"]).admin_level.max()
        )
        gdf = gpd.read_parquet(
            path,
            bbox=block_poly.bounds,
            filters=[("admin_level", "==", max_admin_level)],
        )
        gdf = gdf[gdf.intersects(block_poly)]
        if not gdf.empty:
            iter_data.append((iso3, year, gdf))

    if not iter_data:
        print("No census data found for block", block_key)
        return

    print("Calculating pixel metrics")
    out = []
    for iso3, year, gdf in iter_data:
        shape_values = [(shape, i + 1) for i, shape in enumerate(gdf.geometry)]
        data = coverage_weighted_sums(pop_raster, shape_values, len(gdf))

        loc_results = gdf[["shape_id"]].copy()
        loc_results["iso3"] = iso3
        loc_results["year"] = year
        loc_results["population"] = data
        out.append(loc_results)

    print("Saving results")
    results = pd.concat(out)
    pm_data.save_raw_validation_data(
        results,
        resolution=resolution,
        version=version,
        block_key=block_key,
        time_point=time_point,
    )


@click.command()
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
def pixel_metrics_task(
    block_key: str, time_point: str, resolution: str, version: str, output_dir: str
) -> None:
    pixel_metrics_main(block_key, time_point, resolution, version, output_dir)


@click.command()
@clio.with_time_point(choices=None)
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def metrics(
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    time_points = pm_data.list_gbd_raked_prediction_time_points(resolution, version)
    time_points = [time_point for time_point in time_points if time_point.endswith("q1")]
    if time_point in time_points:
        time_points = [time_point]
    elif time_point is not None:
        msg = (
            f"Time point {time_point} not found in {resolution} {version}.\n"
            f"Valid time points are: {time_points}"
        )
        raise ValueError(msg)

    validation_frame = pm_data.load_validation_frame(resolution)
    block_keys = list(validation_frame.block_key.unique())

    # versions = [f"2025_11_08.0{(i + 1):02d}" for i in range(60)]

    jobmon.run_parallel(
        runner="pmtask validate",
        task_name="pixel_metrics",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "20G",
            "runtime": "10m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "block-key": block_keys,
            # "version": versions,
            "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
            # "time-point": time_point,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("validate_pixel_metrics"),
        concurrency_limit=1_000,
    )
