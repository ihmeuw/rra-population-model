from typing import NamedTuple

import geopandas as gpd
import shapely

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData


class TileMetadata(NamedTuple):
    block_key: str
    key: str
    polygon: shapely.Polygon
    gs: gpd.GeoSeries

    @classmethod
    def from_model_frame(
        cls,
        model_frame: gpd.GeoDataFrame,
        tile_key: str,
    ) -> "TileMetadata":
        tile_frame = model_frame[model_frame.tile_key == tile_key]
        return cls(
            block_key=tile_frame.block_key.iloc[0],
            key=tile_key,
            polygon=tile_frame.geometry.iloc[0],
            gs=tile_frame.geometry,
        )


class TrainingMetadata(NamedTuple):
    model_frame: gpd.GeoDataFrame
    tile_meta: TileMetadata
    tile_neighborhood: list[TileMetadata]
    resolution: str
    time_point: str
    intersecting_admins: gpd.GeoDataFrame
    denominators: list[str]
    features: list[str]


def get_training_metadata(
    tile_meta: TileMetadata,
    model_frame: gpd.GeoDataFrame,
    resolution: str,
    time_point: str,
    intersecting_admins: gpd.GeoDataFrame,
    pm_data: PopulationModelData,
) -> TrainingMetadata:
    try:
        full_shape = intersecting_admins.buffer(0).union_all()
    except shapely.errors.GEOSException:
        # Buffer usually fixes small topological errors, but
        # for at least one tile it causes a GEOS exception.
        # This should be investigated, but just going with something
        # that appears to work for now.
        full_shape = intersecting_admins.union_all()

    overlaps = model_frame.intersects(full_shape)
    neighborhood_keys = model_frame[overlaps].tile_key.tolist()

    tile_neighborhood = [
        TileMetadata.from_model_frame(model_frame, key) for key in neighborhood_keys
    ]

    features = pm_data.list_features(resolution, tile_meta.block_key, time_point)

    return TrainingMetadata(
        tile_meta=tile_meta,
        model_frame=model_frame,
        resolution=resolution,
        time_point=time_point,
        tile_neighborhood=tile_neighborhood,
        intersecting_admins=intersecting_admins,
        denominators=pmc.DENOMINATORS,
        features=features,
    )
