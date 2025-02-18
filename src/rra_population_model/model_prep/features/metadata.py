from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pyproj
import rasterra as rt
import shapely

from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)


class FeatureMetadata(NamedTuple):
    model_frame: gpd.GeoDataFrame
    block_frame: gpd.GeoDataFrame
    working_crs: str | pyproj.CRS
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
        provider="microsoft_v4",
        measure="density",
        resolution=resolution,
        time_point="2023q4",
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
    working_crs: str | pyproj.CRS,
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
    return block_bounds  # type: ignore[no-any-return]


def get_working_crs(block_frame: gpd.GeoDataFrame) -> str | pyproj.CRS:
    """Choose a working CRS based on the extent of the block frame.

    If the block frame crosses the antimeridian, we need to use a special CRS
    that can handle this. Otherwise, we can use the standard equal area CRS.
    """
    xmin, ymin, xmax, ymax = block_frame.to_crs(
        pmc.CRSES["wgs84"].to_pyproj()
    ).total_bounds

    longitude_cutoff = 170
    if xmax < -longitude_cutoff or xmin > longitude_cutoff:
        working_crs = pmc.CRSES["equal_area_anti_meridian"].to_pyproj()
    else:
        working_crs = pmc.CRSES["equal_area"].to_pyproj()
    return working_crs


def get_working_model_frame(
    model_frame: gpd.GeoDataFrame, working_crs: str | pyproj.CRS
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
