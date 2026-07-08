import click
import geopandas as gpd
import numpy as np
import rasterio
import rasterra as rt
import shapely
import tqdm
from affine import Affine
from rasterra._features import raster_geometry_mask
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
        with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHEMAX_MB):
            raster = rt.load_raster(
                pm_data.compiled_prediction_vrt_path(time_point, model_spec, measure="population"),
                load_bounds,
            )
        # mask in place rather than via raster.mask(), which would copy the
        # full country canvas
        outside = raster_geometry_mask(
            data_transform=raster.transform,
            data_width=raster._ndarray.shape[1],
            data_height=raster._ndarray.shape[0],
            shapes=[geometry],
        )[0]
        raster._ndarray[outside] = raster.no_data_value

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
            "memory": "8G",
            "runtime": "6m",
            "project": "proj_rapidresponse",
        },
        max_attempts=4,
        resource_scales={
            "memory":  iter([30     , 100    , 500    ]),  # G
            "runtime": iter([10 * 60, 30 * 60, 30 * 60]),  # seconds
        },
        log_root=pm_data.log_dir("postprocess_admin_rasters"),
    )
