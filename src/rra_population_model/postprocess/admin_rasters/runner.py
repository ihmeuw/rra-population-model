import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterra as rt
import shapely
import tqdm
from affine import Affine
from rasterio.features import rasterize
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

HIERARCHY_VERSION = "gbd_2023"
SHAPES_VERSION = "lsae_1285_a0"
# 2020q1 buildings are just a copy of 2020q2, so we don't save country files for users
EXCLUDED_TIME_POINTS = ["2020q1"]
# geometries are buffered by this many meters when selecting modeling blocks
BUFFER_SIZE = 1000
# countries intersecting blocks within this many block columns of the map edge
# are rebuilt from raked prediction blocks recentered on the antimeridian
BOUNDARY_BLOCKS = 8
# GDAL's block cache defaults to 5% of node RAM; cap it during reads so big
# windowed VRT/COG reads don't inflate peak memory
GDAL_CACHEMAX_MB = 512

# Per-task resources are predicted from each country's bounding-box "canvas":
# the float32 raster spanning its full extent at the model resolution, using
# the antimeridian-recentered bbox where that is narrower. Observed jobmon max
# RSS tracks the canvas tightly for both worker branches (peak <= ~3.5G +
# ~3.0x canvas; runtime <= ~40s + ~23s per canvas GB; calibrated on five
# single-time-point test runs -- see
# .claude/early_access/calibrate_canvas_bins.py). Requests carry a 1.1x memory
# and 2x runtime margin; jobmon's default +50% retry bump is the backstop.
MEMORY_FLOOR_GB = 3.5
MEMORY_PER_CANVAS_GB = 3.0
MEMORY_MARGIN = 1.1
MEMORY_BOUNDS_GB = (5, 480)
RUNTIME_FLOOR_S = 40.0
RUNTIME_PER_CANVAS_S = 23.0
RUNTIME_MARGIN = 2.0
MIN_RUNTIME_MIN = 5


def location_canvas_gb(shapes: gpd.GeoDataFrame, pixel_size: float) -> pd.Series:
    """Bounding-box canvas (GB of float32) per location at the model resolution."""
    parts = shapes[["location_id", "geometry"]].explode(index_parts=False)
    bounds = parts.bounds  # cylindrical equal-area maps a lon/lat bbox to a bbox
    transformer = pyproj.Transformer.from_crs(
        shapes.crs, pmc.CRSES["equal_area"].code, always_xy=True
    )
    min_x, min_y = transformer.transform(bounds.minx.to_numpy(), bounds.miny.to_numpy())
    max_x, max_y = transformer.transform(bounds.maxx.to_numpy(), bounds.maxy.to_numpy())
    world_width = 2 * np.abs(pmc.CRSES["equal_area"].bounds[0])
    # parts crossing the prime meridian span the recentered map edge to edge
    straddles = (min_x < 0) & (max_x >= 0)
    extents = pd.DataFrame({
        "location_id": parts["location_id"].to_numpy(),
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        # the same parts re-centered on the antimeridian
        "min_x_am": np.where(straddles, 0.0, np.where(min_x < 0, min_x + world_width, min_x)),
        "max_x_am": np.where(straddles, world_width, np.where(max_x < 0, max_x + world_width, max_x)),
    }).groupby("location_id").agg(
        min_x=("min_x", "min"), max_x=("max_x", "max"),
        min_y=("min_y", "min"), max_y=("max_y", "max"),
        min_x_am=("min_x_am", "min"), max_x_am=("max_x_am", "max"),
    )
    width = np.minimum(
        extents["max_x"] - extents["min_x"],
        extents["max_x_am"] - extents["min_x_am"],
    )
    height = extents["max_y"] - extents["min_y"]
    return (width / pixel_size) * (height / pixel_size) * 4 / 1024**3


def location_resources(canvas_gb: float) -> dict[str, str]:
    memory_gb = MEMORY_MARGIN * (MEMORY_FLOOR_GB + MEMORY_PER_CANVAS_GB * canvas_gb)
    memory_gb = int(np.ceil(np.clip(memory_gb, *MEMORY_BOUNDS_GB)))
    runtime_min = RUNTIME_MARGIN * (RUNTIME_FLOOR_S + RUNTIME_PER_CANVAS_S * canvas_gb) / 60
    runtime_min = int(np.ceil(max(runtime_min, MIN_RUNTIME_MIN)))
    return {"memory": f"{memory_gb}G", "runtime": f"{runtime_min}m"}


def snap_to_grid(
    bounds: tuple[float, float, float, float],
    grid_x_min: float,
    grid_y_min: float,
    pixel_size: float,
) -> shapely.Polygon:
    """Expand bounds outward to the pixel grid anchored at (grid_x_min, grid_y_min).

    Windowed reads with off-grid bounds produce a raster whose grid is shifted
    sub-pixel relative to the model grid, so read windows must be snapped first.
    """
    x_min, y_min, x_max, y_max = bounds
    return shapely.box(
        grid_x_min + np.floor((x_min - grid_x_min) / pixel_size) * pixel_size,
        grid_y_min + np.floor((y_min - grid_y_min) / pixel_size) * pixel_size,
        grid_x_min + np.ceil((x_max - grid_x_min) / pixel_size) * pixel_size,
        grid_y_min + np.ceil((y_max - grid_y_min) / pixel_size) * pixel_size,
    )


def rasterize_outside_mask(
    raster: rt.RasterArray,
    geometry: shapely.Polygon | shapely.MultiPolygon,
    max_strip_bytes: int = 2**29,
) -> np.ndarray:
    """Boolean mask of cells outside the geometry, rasterized in row strips.

    GDAL burns geometries through a float64 buffer covering the rasterized
    extent (8 bytes/pixel), so one-shot rasterization of a large country
    transiently costs 2x the country canvas; strips bound that.
    """
    height, width = raster._ndarray.shape
    strip_rows = max(1, min(height, max_strip_bytes // (width * 8)))
    outside = np.empty((height, width), dtype=bool)
    t = raster.transform
    for row_start in range(0, height, strip_rows):
        row_stop = min(row_start + strip_rows, height)
        # clip the geometry to the strip first: rasterize() re-converts and
        # re-validates every vertex of the shapes it is given per call, which
        # is ruinously slow for complex coastlines repeated over many strips
        clipped = shapely.clip_by_rect(
            geometry,
            t.c,
            t.f + row_stop * t.e,
            t.c + width * t.a,
            t.f + row_start * t.e,
        )
        # clipping can leave degenerate lines/points where the geometry only
        # touches the strip edge; keep polygons only
        if clipped.geom_type == "GeometryCollection":
            clipped = shapely.union_all(
                [g for g in shapely.get_parts(clipped) if g.geom_type == "Polygon"]
            )
        if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
            outside[row_start:row_stop] = True
            continue
        outside[row_start:row_stop] = rasterize(
            [clipped],
            out_shape=(row_stop - row_start, width),
            transform=t * Affine.translation(0, row_start),
            fill=1,
            default_value=0,
        )
    return outside


def shift_to_antimeridian(raster: rt.RasterArray) -> rt.RasterArray:
    """Shift a global raster by half the world width (no resampling)."""
    if raster.crs != pmc.CRSES["equal_area"].code:
        raise ValueError("Transformation being applied to world cylindrical (equal area) only.")
    shift_px = np.abs(pmc.CRSES["equal_area"].bounds[0])
    shift_decimals = len(str(shift_px).split('.')[-1])
    if shift_decimals != 2:
        raise ValueError(f"Expected 2 decimals in CRS width, got {shift_decimals}")
    target_crs = pmc.CRSES["equal_area_anti_meridian"].to_pyproj()

    # NOTE: rasterra bounds are ordered (x_min, x_max, y_min, y_max)
    if raster.bounds[0] < 0 and raster.bounds[1] > 0:
        raise ValueError("Crosses prime meridian")

    # Update affine transform x origin by half the world width
    t = raster.transform
    if t.c < 0:
        t_c = np.round(shift_px + t.c, shift_decimals)
    else:
        t_c = np.round(t.c - shift_px, shift_decimals)
    new_transform = Affine(t.a, t.b, float(t_c), t.d, t.e, t.f)

    return rt.RasterArray(
        raster.to_numpy(),
        transform=new_transform,
        crs=target_crs,
        no_data_value=raster.no_data_value,
    )


def admin_rasters_main(
    location_id: int,
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
) -> None:
    print("Preparing metadata")
    pm_data = PopulationModelData(output_dir)
    model_spec = pm_data.load_model_specification(resolution, version)
    modeling_frame = pm_data.load_modeling_frame(resolution)

    hierarchy = pm_data.load_gbd_raking_input("hierarchy", HIERARCHY_VERSION)
    ihme_loc_id = hierarchy.set_index('location_id').loc[location_id, 'ihme_loc_id']

    print("Loading and creating buffered geometry")
    shapes = pm_data.load_gbd_raking_input("shapes", SHAPES_VERSION)
    geometry = (
        shapes
        .set_index('location_id')
        .loc[[location_id]]
        .to_crs(pmc.CRSES["equal_area"].code)
        .loc[location_id, 'geometry']
    )
    buffered_geometry = (
        gpd.GeoSeries(geometry)
        .explode(index_parts=True)
        .convex_hull.buffer(BUFFER_SIZE)
        .union_all()
    )

    block_key_x = modeling_frame["block_key"].apply(lambda x: int(x.split("X")[0][-4:]))
    block_key_x_max = block_key_x.max()
    intersects_buffer = modeling_frame.intersects(buffered_geometry)
    modeling_frame = modeling_frame.loc[intersects_buffer]
    block_key_x = block_key_x.loc[intersects_buffer]
    near_antimeridian = (
        (block_key_x <= BOUNDARY_BLOCKS)
        | (block_key_x >= block_key_x_max - BOUNDARY_BLOCKS)
    ).any()

    pixel_size = float(resolution)
    if near_antimeridian:
        print("Loading raked prediction blocks and reprojecting due to antimeridian proximity")
        block_groups = modeling_frame.groupby("block_key")

        raster = []
        for block_key, block_frame in tqdm.tqdm(block_groups, total=len(block_groups)):
            block_x_min, block_y_min, _, _ = block_frame.total_bounds
            overlap = geometry.intersection(shapely.box(*block_frame.total_bounds))
            if overlap.is_empty or overlap.area == 0:
                continue
            subset_bounds = snap_to_grid(overlap.bounds, block_x_min, block_y_min, pixel_size)
            with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHEMAX_MB):
                block_raster = pm_data.load_raked_prediction(
                    block_key, time_point, model_spec,
                    subset_bounds=subset_bounds,
                )
            block_raster = block_raster.clip(overlap).mask(overlap)
            block_raster = shift_to_antimeridian(block_raster)
            raster.append(block_raster)
        raster = rt.merge(raster)
    else:
        print("Loading compiled COGs")
        # tile edges sit on the model grid, so any tile corner anchors the snap
        grid_x_min, grid_y_min, _, _ = modeling_frame.total_bounds
        load_bounds = snap_to_grid(geometry.bounds, grid_x_min, grid_y_min, pixel_size)
        vrt_path = pm_data.compiled_prediction_vrt_path(time_point, model_spec, measure="population")
        with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHEMAX_MB):
            # windowed read directly via rasterio: rt.load_raster's boundless
            # read allocates ~2 extra copies of the window, and the snapped
            # window is always inside the VRT so boundless is unnecessary
            with rasterio.open(vrt_path) as f:
                window = rasterio.windows.from_bounds(*load_bounds.bounds, transform=f.transform)
                raster = rt.RasterArray(
                    f.read(1, window=window),
                    transform=f.window_transform(window),
                    crs=f.crs,
                    no_data_value=f.nodata,
                )
            # mask in place rather than via raster.mask(), which would copy
            # the full country canvas
            outside = rasterize_outside_mask(raster, geometry)
            raster._ndarray[outside] = raster.no_data_value
            del outside

    print("Setting zeros to no data")
    # in place to avoid copying the full country canvas
    raster._ndarray[raster._ndarray == 0] = raster.no_data_value

    print("Saving country raster")
    pm_data.save_country_data(raster, resolution, version, ihme_loc_id, time_point)


@click.command()
@clio.with_location_id()
@clio.with_time_point(choices=None)
@clio.with_resolution()
@clio.with_version()
@clio.with_output_directory(pmc.MODEL_ROOT)
def admin_rasters_task(
    location_id: int,
    time_point: str,
    resolution: str,
    version: str,
    output_dir: str,
) -> None:
    admin_rasters_main(location_id, time_point, resolution, version, output_dir)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def admin_rasters(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    compiled_time_points = [
        tp
        for tp in pm_data.list_compiled_prediction_time_points(
            resolution, version, measure="population"
        )
        if tp not in EXCLUDED_TIME_POINTS
    ]
    time_points = clio.convert_choice(time_point, compiled_time_points)

    hierarchy = pm_data.load_gbd_raking_input("hierarchy", HIERARCHY_VERSION)
    national = hierarchy.loc[hierarchy['level'] == 3, ['location_id', 'ihme_loc_id']]

    shapes = pm_data.load_gbd_raking_input("shapes", SHAPES_VERSION)
    canvas_gb = location_canvas_gb(shapes, float(resolution))
    resources = {
        location_id: location_resources(canvas_gb[location_id])
        for location_id in national['location_id']
    }

    to_run = []
    complete = 0
    for location_id, ihme_loc_id in national.itertuples(index=False):
        for tp in sorted(time_points):
            output_path = pm_data.country_data_path(resolution, version, ihme_loc_id, tp)
            if output_path.exists():
                complete += 1
            else:
                to_run.append((location_id, tp))

    print(f"Running {len(to_run)} location-time points ({complete} already complete).")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="admin_rasters",
        flat_node_args=(("location-id", "time-point"), to_run),
        task_args={
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "project": "proj_rapidresponse",
            "memory": f"{MEMORY_BOUNDS_GB[0]}G",
            "runtime": f"{MIN_RUNTIME_MIN}m",
        },
        per_task_resources=lambda args: resources[args[0]],
        max_attempts=2,
        log_root=pm_data.log_dir("postprocess_admin_rasters"),
    )
