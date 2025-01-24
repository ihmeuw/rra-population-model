from pathlib import Path
from typing import Literal, NamedTuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterra as rt
import shapely
import pyproj
from affine import Affine
from rasterio.fill import fillnodata
from scipy.signal import oaconvolve

from rra_population_model import constants as pmc
from rra_population_model.data import BuildingDensityData, PopulationModelData

#############
# Utilities #
#############

def precise_floor(a: float, precision: int = 0) -> float:
    """Round a number down to a given precision.

    Parameters
    ----------
    a
        The number to round down.
    precision
        The number of decimal places to round down to.

    Returns
    -------
    float
        The rounded down number.
    """
    return float(np.true_divide(np.floor(a * 10**precision), 10**precision))


def suppress_noise(
    raster: rt.RasterArray,
    noise_threshold: float = 0.01,
    fill_value: float = 0.0,
) -> rt.RasterArray:
    """Suppress small values in a raster.

    Parameters
    ----------
    raster
        The raster to suppress noise in.
    noise_threshold
        The threshold below which values are considered noise.

    Returns
    -------
    rt.RasterArray
        The raster with small values suppressed
    """
    raster._ndarray[raster._ndarray < noise_threshold] = fill_value  # noqa: SLF001
    return raster


class FeatureMetadata(NamedTuple):
    model_frame: gpd.GeoDataFrame
    block_frame: gpd.GeoDataFrame
    working_crs: pyproj.CRS
    block_bounds: dict[str, shapely.Polygon]
    block_template: rt.RasterArray
    resolution: str
    block_key: str
    time_point: str

    @property
    def shared_kwargs(self) -> dict[str, str]:
        return {
            "resolution": self.resolution,
            "block_key": self.block_key,
            "time_point": self.time_point,
        }
    

def get_feature_metadata(
    pm_data: PopulationModelData,
    bd_data: BuildingDensityData,
    resolution: str,
    block_key: str,
    time_point: str,
) -> FeatureMetadata:
    
    model_frame = pm_data.load_modeling_frame(resolution)
    block_frame = model_frame[model_frame.block_key == block_key]
    working_crs = get_working_crs(block_frame)
    block_bounds = get_block_bounds(block_frame, model_frame, working_crs)
    block_template = bd_data.load_tile(  # Any provider or measure would do here
        provider="ghsl_r2023a",
        measure="density",
        resolution=resolution,
        time_point="2020q1",
        block_key=block_key,
    )
    return FeatureMetadata(
        model_frame=model_frame,
        block_frame=block_frame,
        working_crs=working_crs,
        block_bounds=block_bounds,
        block_template=block_template,
        resolution=resolution,
        block_key=block_key,
        time_point=time_point,
    )
    


def get_block_bounds(
    block_frame: gpd.GeoDataFrame,
    model_frame: gpd.GeoDataFrame,
    working_crs: pyproj.CRS,
) -> dict[str, shapely.Polygon]:
    model_poly_working = (
        block_frame.dissolve("block_key").to_crs(working_crs).geometry.iloc[0]
    )
    model_frame_working = get_working_model_frame(model_frame, working_crs)
    # Slightly bigger than half the max radius
    buffer_size = max(pmc.FEATURE_AVERAGE_RADII) // 1.9

    overlapping = model_frame_working["valid"] & model_frame_working.intersects(
        model_poly_working.buffer(buffer_size)
    )
    block_bounds = (
        model_frame.loc[overlapping, ["block_key", "geometry"]]
        .dissolve("block_key")
        .geometry.to_dict()
    )
    return block_bounds


def get_working_crs(block_frame: gpd.GeoDataFrame) -> str:
    """Choose a working CRS based on the extent of the block frame.

    If the block frame crosses the antimeridian, we need to use a special CRS
    that can handle this. Otherwise, we can use the standard equal area CRS.
    """
    xmin, ymin, xmax, ymax = block_frame.to_crs(pmc.CRSES["wgs84"].to_pyproj()).total_bounds

    longitude_cutoff = 170
    if xmax < -longitude_cutoff or xmin > longitude_cutoff:
        working_crs = pmc.CRSES["equal_area_anti_meridian"].to_pyproj()
    else:
        working_crs = pmc.CRSES["equal_area"].to_pyproj()
    return working_crs


def get_working_model_frame(
    model_frame: gpd.GeoDataFrame, working_crs: str
) -> gpd.GeoDataFrame:
    model_frame_working = model_frame.to_crs(working_crs)
    # If we're working in an antimeridian CRS, the reprojection will cause any geometries
    # lying on the original prime meridian to break. Specifically, they will end up with
    # a x/longitude range that spans the globe. This is the same problem we're trying
    # to avoid in our working area with the rotation. If we leave these broken geometries
    # in, we will pick up a bunch of false positives when we intersect with our
    # model polygon, so we use a heuristic to filter them out first.
    #
    # Our tiles can have different sizes (they're generally all the same except for the
    # easternmost tiles, which are smaller). We get a representative tile size by
    # taking the median of the x_span as only two columns of tiles will have a different
    # x_span. We'll then check that the x_span of the working frame is at most
    # 1.001 times the representative tile size.
    bounds = model_frame_working.bounds
    x_span = bounds.maxx - bounds.minx
    x_span_median = np.median(x_span) * 1.001
    model_frame_working["valid"] = x_span < x_span_median
    return model_frame_working


def make_smoothing_convolution_kernel(
    pixel_resolution_m: int | float,
    radius_m: int | float,
    kernel_type: Literal["uniform", "gaussian"] = "uniform",
) -> npt.NDArray[np.float64]:
    """Make a convolution kernel for spatial averaging/smoothing.

    A convolution kernel is a (relatively) small matrix that is used to apply a
    localized transformation to a raster. Here we are choosing a kernel whose
    values are all positive and sum to 1 (thus representing a probability mass
    function). This special property means that the kernel can be used to
    compute a weighted average of the pixels in a neighborhood of a given pixel.
    In image processing, this is often used to smooth out noise in the image or
    to blur the image.

    This function produces both uniform and gaussian kernels. A uniform kernel
    is a circle with equal weights for all pixels inside the circle. A gaussian
    kernel is a circle with a gaussian distribution of weights, i.e. the weights
    decrease as you move away from the center of the circle.

    Parameters
    ----------
    pixel_resolution_m
        The resolution of the raster in meters.
    radius_m
        The radius of the kernel in meters.
    kernel_type
        The type of kernel to make. Either "uniform" or "gaussian".
        A uniform kernel is a circle with equal weights for all pixels inside
        the circle. A gaussian kernel is a circle with a gaussian distribution
        of weights.

    Returns
    -------
    np.ndarray
        The convolution kernel.
    """
    radius = int(radius_m // pixel_resolution_m)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    if kernel_type == "uniform":
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        mask = x**2 + y**2 < radius**2
        kernel[mask] = 1 / np.sum(mask)
    elif kernel_type == "gaussian":
        kernel = np.exp(-(x**2 + y**2) / (radius**2))
        kernel = kernel / kernel.sum()
    else:
        raise NotImplementedError
    return kernel


def make_spatial_average(
    tile: rt.RasterArray,
    radius: int | float,
    kernel_type: Literal["uniform", "gaussian"] = "uniform",
) -> rt.RasterArray:
    """Compute a spatial average of a raster.

    Parameters
    ----------
    tile
        The raster to average.
    radius
        The radius of the averaging kernel in meters.
    kernel_type
        The type of kernel to use. Either "uniform" or "gaussian".

    Returns
    -------
    rt.RasterArray
        A raster with the same extent and resolution as the input raster, but
        with the values replaced by the spatial average of the input raster.
        Note that pixels within 1/2 the radius of the edge of the raster will
        have a reduced number of contributing pixels, and thus will be less
        accurate. See the documentation for scipy.signal.oaconvolve for more details.
    """
    arr = np.nan_to_num(tile.to_numpy())

    kernel = make_smoothing_convolution_kernel(tile.x_resolution, radius, kernel_type)

    out_image = oaconvolve(arr, kernel, mode="same")
    # TODO: Figure out why I did this
    out_image -= np.nanmin(out_image)
    min_value = 0.005
    out_image[out_image < min_value] = 0.0

    out_image = out_image.reshape(arr.shape)
    out_raster = rt.RasterArray(
        out_image,
        transform=tile.transform,
        crs=tile.crs,
        no_data_value=tile.no_data_value,
    )
    return out_raster


############
# Features #
############


def load_and_format_building_density(
    bd_data: BuildingDensityData,
    time_point: str,
    tile_keys: list[str],
    raster_template: rt.RasterArray,
) -> tuple[rt.RasterArray, rt.RasterArray]:
    def _load_clean_tile(tile_key: str) -> rt.RasterArray:
        tile = bd_data.load_building_density_tile(time_point, tile_key)  # type: ignore[attr-defined]
        # The resolution of the MSFT tiles has too many decimal points.
        # This causes tiles slightly west of the antimeridian to cross
        # over and really mucks up reprojection. We'll clip the values
        # here to 5 decimal places (ie to 100 microns), explicitly
        # rounding down. This reduces the width of the tile by
        # 512*0.0001 = 0.05m or 50cm, enough to fix roundoff issues.
        x_res, y_res = tile.resolution
        xmin, xmax, ymin, ymax = tile.bounds
        tile._transform = Affine(  # noqa: SLF001
            a=precise_floor(x_res, 4),
            b=0.0,
            c=xmin,
            d=0.0,
            e=-precise_floor(-y_res, 4),
            f=ymax,
        )
        reprojected_tile = tile.reproject(
            dst_resolution=raster_template.x_resolution,
            dst_crs=raster_template.crs,
            resampling="average",
        )
        return reprojected_tile  # type: ignore[no-any-return]

    raw_tiles = [_load_clean_tile(tile_key) for tile_key in tile_keys]
    raw_building_density = rt.merge(raw_tiles)
    buffered_building_density = suppress_noise(raw_building_density)
    building_density = buffered_building_density.resample_to(raster_template, "average")
    building_density = suppress_noise(building_density)
    return building_density, buffered_building_density


def _load_and_process_ntl(
    ntl_path: Path,
    feature_metadata: FeatureMetadata,
) -> rt.RasterArray:
    block_gdf = feature_metadata.block_frame
    threshold = 100  # Degrees, just need something bigger than the block

    wgs84 = pmc.CRSES["wgs84"].to_pyproj()
    clip_gdf = block_gdf.dissolve("block_key").to_crs(wgs84)
    xmin, ymin, xmax, ymax = clip_gdf.total_bounds

    if (
        not clip_gdf.is_valid.all()
        or np.isinf(clip_gdf.total_bounds).any()
        or xmax - xmin > threshold
    ):
        bounds = block_gdf.bounds
        xmin, ymin, xmax, ymax = (
            bounds["minx"],
            bounds["miny"],
            bounds["maxx"],
            bounds["maxy"],
        )

        lat_long_bounds = block_gdf.to_crs(wgs84).bounds

        xs_min, xs_max = lat_long_bounds[xmin == xmin.min()].iloc[0][["minx", "maxx"]]
        xe_min, xe_max = lat_long_bounds[xmax == xmax.max()].iloc[0][["minx", "maxx"]]

        if xs_max - xs_min > threshold:  # Left edge ran over
            xs, xe = -180.0, xe_max
        elif xe_max - xe_min > threshold:  # Right edge ran over
            xs, xe = xs_min, 180.0
        else:
            # Sometimes projecting the tiles individually just fixes things
            if (xe_max - xs_min) > threshold:
                msg = "Unknown projection issue"
                raise RuntimeError(msg)
            xs, xe = xs_min, xe_max

        ys = lat_long_bounds[ymin == ymin.min()].miny.iloc[0]
        ye = lat_long_bounds[ymax == ymax.max()].maxy.iloc[0]

        lat_bound, lon_bound = 90, 180
        bad_lon = not (-lon_bound <= xs < xe <= lon_bound) or xe - xs >= threshold
        bad_lat = not (-lat_bound <= ys < ye <= lat_bound) or ye - ys >= threshold
        if bad_lon or bad_lat:
            msg = "Unknown projection issue"
            raise RuntimeError(msg)

        clip_poly = shapely.box(xs, ys, xe, ye)
    else:
        clip_poly = clip_gdf.geometry.iloc[0]

    ntl = (
        rt.load_raster(ntl_path, clip_poly)
        .set_no_data_value(np.nan)
        .resample_to(feature_metadata.block_template, "average")
        .astype(np.float32)
    )
    # The high resolution NTL data in particular has some no data values that need
    # to be filled in, though I have not systematically checked this for all NTL data.
    filled_ntl_data = fillnodata(
        ntl.to_numpy(),
        mask=~ntl.no_data_mask,
    )
    filled_ntl = rt.RasterArray(
        np.nan_to_num(filled_ntl_data),
        transform=ntl.transform,
        crs=ntl.crs,
        no_data_value=ntl.no_data_value,
    )

    return filled_ntl


def load_and_format_ntl(
    feature_metadata: FeatureMetadata,
) -> rt.RasterArray:
    year = int(feature_metadata.time_point.split("q")[0])
    year = min(year, 2022)
    year = max(year, 2000)
    if year < 2021:  # noqa: PLR2004
        rel_path = f"57_dmsp_viirs_harm_2/dataverse_files/LongNTL_{year}.tif"
    elif year < 2023:  # noqa: PLR2004
        rel_path = f"57_dmsp_viirs_harm_2/v4_update/LongNTL_{year}.tif"
    else:
        msg = f"No NTL for year {year}"
        raise NotImplementedError(msg)
    path = pmc.GEOSPATIAL_COVARIATES_ROOT / rel_path
    return _load_and_process_ntl(path, feature_metadata)
