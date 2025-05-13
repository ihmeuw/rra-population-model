from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.model_prep.features.metadata import get_feature_metadata

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from pathlib import Path

import rasterio
from rasterio.crs import CRS
import rasterio.mask
import rasterio.plot

from scipy.ndimage import distance_transform_edt
from affine import Affine
import matplotlib.pyplot as plt

from shapely.geometry import box, Point, LineString
from shapely.geometry.base import BaseGeometry


from pathlib import Path
from typing import Tuple, Union, Any

def read_and_clip_vector(
    vector_file_path: Union[str, Path],
    bbox: BaseGeometry
) -> gpd.GeoDataFrame:

    # Read the vector data
    vector_gdf = gpd.read_parquet(vector_file_path, bbox=bbox)

    return vector_gdf


def expand_block_bounding_box(modeling_frame: gpd.GeoDataFrame, block_key: str, expansion_factor: float = 1.0) -> gpd.GeoSeries:
    # Subset the frame to the block
    block_tiles = modeling_frame[modeling_frame["block_key"] == block_key]

    # Compute total bounding box for the block
    total_bounds = block_tiles.total_bounds  # returns [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = total_bounds

    # Compute expansions
    height_exp = (maxy - miny) * expansion_factor
    width_exp = (maxx - minx) * expansion_factor

    # Apply expansion
    new_minx = minx - width_exp
    new_maxx = maxx + width_exp
    new_miny = miny - height_exp
    new_maxy = maxy + height_exp

    return box(new_minx, new_miny, new_maxx, new_maxy)


# Get Meta data from TIF file
def get_metadata(tif_file_path: Path) -> Tuple[dict, Tuple[int, int], Affine, CRS]:
    
    with rasterio.open(tif_file_path) as src:
        out_meta = src.meta
        shape = src.shape
    
    # Transform Data
    transform = out_meta['transform']
    # Out metadata is exactly the same as our in metadata.
    target_crs = out_meta['crs']
    
    return out_meta, shape, transform, target_crs


# Generate expanded shape and new transform 
def expand_shape_and_transform(original_shape: Tuple[int, int], original_transform: Affine) -> Tuple[Tuple[int, int], Affine]:

    # New shape
    new_shape = (original_shape[0]*3, original_shape[1]*3)

    # New transform
    # Calculate the adjustment considering the resolution
    height = original_shape[0]
    width = original_shape[1]
    
    resolution = original_transform.a
    adjustment_x = resolution * width  # Width adjustment (to the left)
    adjustment_y = resolution * height  # Height adjustment (up)

    # Update the transform by adding adjustments
    new_transform = Affine(original_transform.a, original_transform.b, original_transform.c - adjustment_x,
                        original_transform.d, original_transform.e, original_transform.f + adjustment_y)

    return new_shape, new_transform 


# Rasterize geometeries found in a GDF onto an empty array based on an existing transform
def rasterize(gdf: GeoDataFrame, out_shape: Tuple[int, int], transform: Affine) -> np.ndarray:

    # Rasterize the GeoDataFrame
    rasterized = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8
    )

    # Switch 1s and 0s
    rasterized = np.logical_not(rasterized).astype(int)
    return rasterized

# Subset expanded distance array to original size
def subset_array(original_shape: Tuple[int, int], computed_array: np.ndarray) -> np.ndarray:

    # Create rows and columns
    start_row = original_shape[0]
    end_row = original_shape[0]*2
    start_col = original_shape[1]
    end_col = original_shape[1]*2

    # Subset original array from expanded distance array
    original_out = computed_array[start_row:end_row, start_col:end_col]

    return original_out



def generate_distance_raster(
    pm_data,
    bd_data,
    resolution: str,
    block_key: str,
    time_point: str,
    overture_class: str,
    overture_type: str,
    output_dir: Path
) -> None:
    overture_root = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/covariates/overture/") # UPDATE NEEDED

    # Step 1: Get feature metadata and modeling frame
    feature_metadata = get_feature_metadata(pm_data, bd_data, resolution, block_key, time_point)
    modeling_frame = feature_metadata[0]
    modeling_frame = modeling_frame.to_crs("EPSG:4326")  # Reproject to WGS84

    # Step 2: Get expanded bounding box
    expanded_bbox = expand_block_bounding_box(modeling_frame, block_key, expansion_factor=1.0).bounds

    # Step 3: Read in vector file and reproject
    vector_file_path = overture_root / overture_type / f"{overture_class}.parquet"
    vector_gdf_subset = read_and_clip_vector(vector_file_path, expanded_bbox)

    # Step 4: Read raster metadata
    raster_template_path = Path(
        f"/mnt/team/rapidresponse/pub/population-model/ihmepop_results/2025_03_22/{time_point}/{block_key}.tif"
    )
    out_meta, shape, transform, target_crs = get_metadata(raster_template_path)

    # Step 5: Generate new shape and transform
    new_shape, new_transform = expand_shape_and_transform(shape, transform)

    # Step 6: Reproject everything to target CRS
    modeling_frame = modeling_frame.to_crs(target_crs)
    vector_gdf_subset = vector_gdf_subset.to_crs(target_crs)

    # Step 7: Rasterize vector geometries
    rasterized = rasterize(vector_gdf_subset, new_shape, new_transform)

    # Step 8: Compute Euclidean distance
    distance_array = distance_transform_edt(rasterized)

    # Step 9: Subset distance array to original raster shape
    original_out = subset_array(shape, distance_array)

    # Step 10: Write distance raster to disk
    original_out = original_out.astype(out_meta['dtype'])
    out_path = Path(output_dir) / f"{block_key}_distance.tif"

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(original_out.reshape((1, *original_out.shape)))

    print(f"Distance raster saved to: {out_path}")
