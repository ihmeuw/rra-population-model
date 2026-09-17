from typing import Any, cast

import geopandas as gpd
import numpy as np
import rasterio
import rasterra as rt
from affine import Affine
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio.fill import fillnodata
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model_prep.features.metadata import FeatureMetadata

# ADD OVERTURE YEAR
TIME_POINT_OVERTURE = "2020q2"
# Number of blocks of buffer around each block. The vector query, the
# rasterization window, and the maximum reliable distance are all derived from
# this single value so they stay consistent.
BLOCK_BUFFER = 1


def read_and_clip_vector(
    overture_class: str,
    overture_type: str,
    bbox: BaseGeometry,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame | None:
    """
    Read and clip an Overture vector parquet file to a bounding box.
    """

    overture_root = pmc.POPULATION_COVARIATE_ROOT / "overture"

    vector_file_path = overture_root / overture_class / f"{overture_type}.parquet"

    gdf = gpd.read_parquet(vector_file_path, bbox=bbox)
    if gdf.empty:
        return None

    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def expand_tile_bounding_box(
    model_frame: gpd.GeoDataFrame,
    block_key: str,
    expansion_factor: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Expand block frame to include neighboring blocks in order to capture features that may be near the edges.
    """
    block = model_frame[model_frame["block_key"] == block_key]
    block = block.to_crs("EPSG:4326")  # Match shapefile CRS

    minx, miny, maxx, maxy = block.total_bounds
    height_exp = (maxy - miny) * expansion_factor
    width_exp = (maxx - minx) * expansion_factor

    bbox_poly = box(
        minx - width_exp,
        miny - height_exp,
        maxx + width_exp,
        maxy + height_exp,
    )
    return cast(tuple[float, float, float, float], bbox_poly.bounds)


def get_metadata_from_block_template(
    block_template: rt.RasterArray,
) -> tuple[tuple[int, int], Affine]:
    """
    Return raster metadata (shape, transform) from a RasterArray template.
    """
    height, width = block_template.shape
    transform: Affine = block_template.transform
    return (height, width), transform


# Generate expanded shape and new transform
def expand_shape_and_transform(
    original_shape: tuple[int, int],
    original_transform: Affine,
    buffer: int,
) -> tuple[tuple[int, int], Affine]:
    # Surround the block with `buffer` blocks on every side.
    factor = 1 + 2 * buffer
    new_shape = (original_shape[0] * factor, original_shape[1] * factor)

    # New transform
    # Calculate the adjustment considering the resolution
    height = original_shape[0]
    width = original_shape[1]

    resolution = original_transform.a
    adjustment_x = resolution * width * buffer  # Width adjustment (to the left)
    adjustment_y = resolution * height * buffer  # Height adjustment (up)

    # Update the transform by adding adjustments
    new_transform = Affine(
        original_transform.a,
        original_transform.b,
        original_transform.c - adjustment_x,
        original_transform.d,
        original_transform.e,
        original_transform.f + adjustment_y,
    )

    return new_shape, new_transform


# Rasterize geometeries found in a GDF onto an empty array based on an existing transform
def rasterize(
    gdf: GeoDataFrame, out_shape: tuple[int, int], transform: Affine
) -> NDArray[np.int_]:
    # Rasterize the GeoDataFrame
    rasterized = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8,
    )

    # Switch 1s and 0s
    rasterized = np.logical_not(rasterized).astype(int)
    return cast(NDArray[np.int_], rasterized)


def clip_to_block_template(
    array: np.ndarray[tuple[Any, ...], Any],
    feature_metadata: FeatureMetadata,
) -> np.ndarray[tuple[Any, ...], Any]:
    """
    Clip the feature array to the block template pixels that are valid (not no_data).
    """
    block_template = feature_metadata.block_template
    mask = ~np.isnan(block_template.to_numpy())

    # Apply the mask
    clipped_array = np.where(mask, array, np.nan)
    return clipped_array


def compute_feature_array(
    feature_metadata: FeatureMetadata,
    rasterized: np.ndarray[tuple[Any, ...], Any],
    mode: str,
    original_shape: tuple[int, int],
    transform: Affine,
    buffer: int,
    bandwidth_m: float = 300.0,
) -> np.ndarray[tuple[Any, ...], Any]:
    """
    Compute feature raster from a rasterized geometry, either distance or KDE density.
    Return original array shape.
    """
    pixel_size = abs(transform.a)

    if mode == "distance":
        # distance expects features=0, background=1
        # Distances are only reliable out to the buffer width; beyond that a
        # nearer feature could exist just outside the frame, so cap there.
        max_distance = buffer * min(original_shape) * pixel_size
        if (rasterized == 0).any():
            result_array = distance_transform_edt(rasterized) * pixel_size
            result_array = np.minimum(result_array, max_distance)
        else:
            # No features in the window: every pixel is at least `max_distance`
            # away, which is exactly the cap we would apply anyway.
            result_array = np.full(rasterized.shape, max_distance, dtype=float)

    elif mode == "kde_density":
        # KDE expects features=1, background=0
        mask = np.logical_not(rasterized).astype(float)
        sigma = bandwidth_m / pixel_size
        result_array = gaussian_filter(mask, sigma=sigma)

    # Subset to original shape (the center block within the buffered window)
    start_row = original_shape[0] * buffer
    end_row = original_shape[0] * (buffer + 1)
    start_col = original_shape[1] * buffer
    end_col = original_shape[1] * (buffer + 1)
    result_array = result_array[start_row:end_row, start_col:end_col]

    clipped_result_array = clip_to_block_template(result_array, feature_metadata)

    return clipped_result_array


def save_results(
    pm_data: Any,
    array: np.ndarray[Any, Any] | rt.RasterArray,
    feature_name: str,
    shared_kwargs: dict[str, Any],
) -> None:
    # Save original array
    pm_data.save_feature(array, feature_name=feature_name, **shared_kwargs)

    # Compute log(1 + x) safely depending on type
    if isinstance(array, rt.RasterArray):
        log_array = rt.RasterArray(
            np.log1p(array.to_numpy()),  # log1p avoids log(0)
            transform=array.transform,
            crs=array.crs,
            no_data_value=array.no_data_value,
        )
    else:
        log_array = np.log1p(array)

    # Save log-transformed version
    pm_data.save_feature(log_array, feature_name=f"log_{feature_name}", **shared_kwargs)


def generate_overture_features(
    pm_data: PopulationModelData,
    feature_metadata: FeatureMetadata,
    overture_class: str,
    overture_type: str,
    mode: str = "distance",  # "distance" or "kde_density"
) -> None:
    all_time_points = pmc.ALL_TIME_POINTS

    # Step 1: Extract metadata
    model_frame = feature_metadata.model_frame
    block_key = feature_metadata.block_key

    # Step 2: Expanded bounding box
    expanded_bbox = expand_tile_bounding_box(
        model_frame, block_key,
        expansion_factor=BLOCK_BUFFER,
    )

    # Read vector file
    vector_gdf_subset = read_and_clip_vector(
        overture_class,
        overture_type,
        expanded_bbox,
        target_crs=model_frame.crs,
    )

    # Get raster metadata
    shape, transform = get_metadata_from_block_template(feature_metadata.block_template)

    # Expand shape and transform for edge effects
    new_shape, new_transform = expand_shape_and_transform(
        shape, transform, buffer=BLOCK_BUFFER
    )

    # Rasterization. When no features fall in the buffered window the query
    # returns None; represent that as an all-background raster (features=0).
    if vector_gdf_subset is None:
        rasterized = np.ones(new_shape, dtype=int)
    else:
        rasterized = rasterize(vector_gdf_subset, new_shape, new_transform)

    # Generate feature array
    feature_array = compute_feature_array(
        feature_metadata,
        rasterized,
        mode,
        original_shape=shape,
        transform=new_transform,
        buffer=BLOCK_BUFFER,
    )

    # Convert feature array → RasterArray first (same shape/transform as template)
    feature_raster = rt.RasterArray(
        feature_array.astype(np.float32),
        transform=transform,
        crs=feature_metadata.block_template.crs,
        no_data_value=np.nan,
    )

    # Fill in nodata pixels based on the mask
    filled_feature_data = fillnodata(
        feature_raster.to_numpy(),
        mask=~feature_raster.no_data_mask,  # valid pixels = ~no_data_mask
    )

    # Convert back to RasterArray with same metadata
    filled_feature_raster = rt.RasterArray(
        # np.nan_to_num(filled_feature_data),
        filled_feature_data,
        transform=feature_raster.transform,
        crs=feature_raster.crs,
        no_data_value=feature_raster.no_data_value,
    )

    if overture_class == "roads":
        prefix = "or"
    elif overture_class == "water":
        prefix = "ow"

    feature_name = f"{prefix}_{overture_type}_{mode}"

    # Save results and log results
    save_results(
        pm_data,
        filled_feature_raster,
        feature_name=feature_name,
        shared_kwargs=feature_metadata.shared_kwargs,
    )

    # Link feature to all time_points
    feature_path = pm_data.feature_path(
        resolution=feature_metadata.resolution,
        block_key=feature_metadata.block_key,
        feature_name=feature_name,
        time_point=feature_metadata.time_point,
    )

    for time_point in all_time_points:
        if time_point == feature_metadata.time_point:
            continue
        pm_data.link_feature(
            source_path=feature_path,
            feature_name=feature_name,
            time_point=time_point,
            block_key=feature_metadata.block_key,
            resolution=feature_metadata.resolution,
        )
        pm_data.link_feature(
            source_path=feature_path,
            feature_name=f"log_{feature_name}",
            time_point=time_point,
            block_key=feature_metadata.block_key,
            resolution=feature_metadata.resolution,
        )

def process_overture(
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
) -> None:
    # Check if time_point is valid
    if feature_metadata.time_point != TIME_POINT_OVERTURE:
        return

    overture_dict = pm_data.list_overture_covariates()

    for overture_class, overture_types in overture_dict.items():
        print(overture_class)
        for overture_type in overture_types:
            print(f'    {overture_type}')
            # Always generate distance
            generate_overture_features(
                pm_data,
                feature_metadata,
                overture_class,
                overture_type,
                mode="distance",
            )

            # If overture is water (or stream_water), also generate density
            if overture_class == "water" and overture_type == "stream_water":
                generate_overture_features(
                    pm_data,
                    feature_metadata,
                    overture_class,
                    overture_type,
                    mode="kde_density",
                )
