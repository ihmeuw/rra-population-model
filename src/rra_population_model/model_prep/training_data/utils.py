from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
import tqdm

from rra_population_model.data import PopulationModelData
from rra_population_model.model_prep.training_data.metadata import (
    TileMetadata,
    TrainingMetadata,
)


def get_intersecting_admins(
    tile_meta: TileMetadata,
    iso3_list: str,
    time_point: str,
    pm_data: PopulationModelData,
) -> gpd.GeoDataFrame:
    year = time_point.split("q")[0]

    admin_data = []
    for iso in iso3_list.split(","):
        a = pm_data.load_census_data(iso, year, tile_meta.polygon)
        # Need to intersect again with the tile poly because we load based on the
        # intersection with the bounding box.
        is_max_admin = a.admin_level == a.admin_level.max()
        intersects_tile = a.intersects(tile_meta.polygon)
        a = a.loc[is_max_admin & intersects_tile]
        admin_data.append(a)

    admins = pd.concat(admin_data, ignore_index=True)
    admins = admins.rename(
        columns={
            "shape_id": "admin_id",
            "population_total": "admin_population",
        }
    )
    admins["admin_area"] = admins.area
    admins["geometry"] = admins.buffer(0)
    return admins.loc[:, ["admin_id", "admin_population", "admin_area", "geometry"]]


def get_training_locations_and_years(
    pm_data: PopulationModelData,
) -> list[tuple[str, str, str]]:
    """Get the locations and years for which we have training data."""
    available_census_years = pm_data.list_census_data()  # noqa: F841
    return [
        ("MEX", "2020", "1"),
        ("USA", "2020", "1"),
    ]


def build_arg_list(
    resolution: str,
    pm_data: PopulationModelData,
    buffer_size: int | float = 5000,
) -> list[tuple[str, str, str]]:
    modeling_frame = pm_data.load_modeling_frame(resolution)
    training_census_years = get_training_locations_and_years(pm_data)

    tile_keys_and_times = defaultdict(list)
    for iso3, year, quarter in training_census_years:
        print(f"Processing {iso3} {year}q{quarter}")
        shape = pm_data.load_census_data(iso3, year)
        a1 = (
            shape.loc[shape.admin_level == 1]
            .explode(index_parts=True)
            .convex_hull.buffer(buffer_size)
            .union_all()
        )
        a1_intersection = modeling_frame[modeling_frame.intersects(a1)]
        for tile_key in a1_intersection.tile_key.unique():
            tile_keys_and_times[(tile_key, f"{year}q{quarter}")].append(iso3)

    to_run = [
        (tile_key, time_point, ",".join(iso3s))
        for (tile_key, time_point), iso3s in tile_keys_and_times.items()
    ]
    return to_run


def build_summary_people_per_structure(
    pm_data: PopulationModelData,
    resolution: str,
) -> pd.DataFrame:
    tile_dirs = list(pm_data.tile_training_data_root(resolution).iterdir())
    data = pd.concat(
        [
            pm_data.load_people_per_structure(resolution, tile_dir.name)
            for tile_dir in tqdm.tqdm(tile_dirs)
        ],
        ignore_index=True,
    )
    return data


def safe_divide(
    a: npt.NDArray[np.floating[Any]] | pd.DataFrame,
    b: npt.NDArray[np.floating[Any]] | pd.DataFrame,
) -> npt.NDArray[np.floating[Any]]:
    """Divide two arrays, but return 0 where both arrays are 0."""

    if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(  # type: ignore[arg-type]
        b.dtype,  # type: ignore[arg-type]
        np.floating,
    ):
        msg = "Both arrays must be floating point."
        raise TypeError(msg)

    mask = ~((a == 0) & (b == 0))
    r = np.zeros_like(a)
    r[mask] = a[mask] / b[mask]
    return r


def get_tile_feature_gdf(
    tile_meta: TileMetadata,
    training_meta: TrainingMetadata,
    pm_data: PopulationModelData,
) -> gpd.GeoDataFrame:
    """Load the raster features for the tile and convert to a GeoDataFrame."""
    kwargs = {
        "resolution": training_meta.resolution,
        "block_key": tile_meta.block_key,
        "time_point": training_meta.time_point,
    }

    tile_features = {}
    for feature_name in training_meta.features:
        tile_features[feature_name] = pm_data.load_feature(
            feature_name=feature_name,
            subset_bounds=tile_meta.polygon,
            **kwargs,
        )

    default_raster = training_meta.denominators[0]
    bd_raster = tile_features.pop(default_raster)
    feature_gdf = (
        bd_raster.to_gdf()
        .reset_index()
        .rename(columns={"value": f"pixel_{default_raster}", "index": "pixel_id"})
        .sort_values("pixel_id")
    )
    feature_gdf["pixel_area"] = feature_gdf.area
    feature_gdf["block_key"] = tile_meta.block_key
    feature_gdf["tile_key"] = tile_meta.key
    feature_gdf["time_point"] = training_meta.time_point

    for feature_name, feature_raster in tile_features.items():
        feature_gdf[f"pixel_{feature_name}"] = feature_raster.to_numpy().flatten()

    tile_gdf = (
        training_meta.intersecting_admins
        # First we just want to subset the admins as the computational complexity
        # of overlay is proportional to the area of the two gdfs (or to the points
        # it has to figure out).
        .overlay(
            tile_meta.gs.buffer(10).to_frame(),
            how="intersection",
            keep_geom_type=True,
        ).overlay(
            feature_gdf,
            how="intersection",
            keep_geom_type=True,
        )
    )
    tile_gdf["isection_area"] = tile_gdf.area
    return tile_gdf


def process_model_gdf(
    model_gdf: gpd.GeoDataFrame,
    training_meta: TrainingMetadata,
) -> gpd.GeoDataFrame:
    """Calculate admin and pixel level properties from an intersection GeoDataFrame.

    Parameters
    ----------
    model_gdf
        The GeoDataFrame of admin/pixel/intersection properties.
    features
        The list of features beyond population, occupancy rate, and building
        density to process.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame with all admin and pixel level properties.
    """
    min_density = 0.01
    pixel_area = model_gdf["pixel_area"].iloc[0]
    min_admin_density = min_density * pixel_area

    model_gdf["admin_population_density"] = (
        model_gdf["admin_population"] / model_gdf["admin_area"]
    )
    model_gdf["admin_area_weight"] = (
        model_gdf["isection_area"] / model_gdf["admin_area"]
    )
    model_gdf["pixel_area_weight"] = (
        model_gdf["isection_area"] / model_gdf["pixel_area"]
    )

    model_gdf["admin_multi_tile"] = (
        model_gdf.groupby("admin_id")["tile_key"].transform("nunique") > 1
    ).astype(float)
    model_gdf["pixel_multi_tile"] = model_gdf["admin_multi_tile"]

    for denominator in training_meta.denominators:
        denominator_df = model_gdf[
            ["tile_key", "pixel_id", "admin_id", "admin_population"]
        ].copy()
        denominator_df["isection_built"] = (
            model_gdf[f"pixel_{denominator}"] * model_gdf["isection_area"]
        )
        denominator_df["pixel_built"] = (
            model_gdf[f"pixel_{denominator}"] * model_gdf["pixel_area"]
        )
        denominator_df["admin_built"] = denominator_df.groupby("admin_id")[
            "isection_built"
        ].transform("sum")
        if denominator[:4] == "msft":
            low_density = denominator_df["admin_built"] < min_admin_density
            denominator_df.loc[low_density, "admin_built"] = 0.0
            denominator_df.loc[low_density, "isection_built"] = 0.0

        denominator_df[f"admin_{denominator}"] = safe_divide(
            denominator_df["admin_built"], model_gdf["admin_area"]
        )

        denominator_df["admin_built_weight"] = safe_divide(
            denominator_df["isection_built"], denominator_df["admin_built"]
        )
        denominator_df["pixel_built_weight"] = safe_divide(
            denominator_df["isection_built"], denominator_df["pixel_built"]
        )

        denominator_df["isection_population"] = (
            denominator_df["admin_population"] * denominator_df["admin_built_weight"]
        )
        denominator_df["pixel_population"] = denominator_df.groupby(
            ["tile_key", "pixel_id"]
        )["isection_population"].transform("sum")

        mask = ~(
            (denominator_df["admin_population"] > 0)
            & (denominator_df["admin_built"] == 0)
        )

        denominator_df["admin_occupancy_rate"] = -1.0
        occupancy_rate = safe_divide(
            denominator_df["admin_population"].astype(float),
            denominator_df["admin_built"],
        )

        denominator_df.loc[mask, "admin_occupancy_rate"] = occupancy_rate[mask]

        denominator_df["admin_log_occupancy_rate"] = -1.0
        denominator_df.loc[mask, "admin_log_occupancy_rate"] = np.log(
            1 + denominator_df.loc[mask, "admin_occupancy_rate"]
        )

        denominator_df["pixel_occupancy_rate"] = -1.0
        denominator_df.loc[mask, "pixel_occupancy_rate"] = safe_divide(
            denominator_df.loc[mask, "pixel_population"],
            denominator_df.loc[mask, "pixel_built"],
        )
        denominator_df["pixel_log_occupancy_rate"] = -1.0
        denominator_df.loc[mask, "pixel_log_occupancy_rate"] = np.log(
            1 + denominator_df.loc[mask, "pixel_occupancy_rate"]
        )

        keep_measures = [
            "admin_built",
            "admin_occupancy_rate",
            "admin_log_occupancy_rate",
            "pixel_population",
            "pixel_occupancy_rate",
            "pixel_log_occupancy_rate",
        ]

        for measure in keep_measures:
            model_gdf[f"{measure}_{denominator}"] = denominator_df[measure]

    for feature in training_meta.features:
        model_gdf[f"admin_{feature}"] = (
            model_gdf[f"pixel_{feature}"] * model_gdf["admin_area_weight"]
        )
        model_gdf[f"admin_{feature}"] = model_gdf.groupby("admin_id")[
            f"admin_{feature}"
        ].transform("sum")

    return model_gdf


def filter_to_admin_gdf(
    model_gdf: gpd.GeoDataFrame,
    training_meta: TrainingMetadata,
) -> gpd.GeoDataFrame:
    """Filter the model GDF to only the admin-level features and rows."""
    keep_cols = (
        ["block_key", "tile_key", "time_point"]
        + [c for c in model_gdf if c[:5] == "admin"]
        + ["geometry"]
    )
    model_gdf = model_gdf.loc[:, keep_cols].groupby("admin_id").first().reset_index()
    model_gdf["geometry"] = (
        training_meta.intersecting_admins.set_index("admin_id")
        .loc[model_gdf["admin_id"], "geometry"]
        .to_numpy()
    )
    return model_gdf


def raster_from_pixel_feature(
    tile_gdf: gpd.GeoDataFrame,
    feature_name: str,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    """Create a raster from a pixel feature in the tile GeoDataFrame.

    Parameters
    ----------
    tile_gdf
        The GeoDataFrame of the pixel features.
    feature_name
        The name of the feature to convert to a raster.
    raster_template
        The template raster to use for the output raster.

    Returns
    -------
    rt.RasterArray
        The raster of the pixel feature.
    """
    idx = np.arange(raster_template.size)
    feature_data = (
        tile_gdf.groupby("pixel_id")[f"pixel_{feature_name}"]
        .first()
        .reindex(idx, fill_value=0.0)
        .to_numpy()
        .astype(np.float32)
        .reshape(raster_template.shape)
    )
    feature_raster = rt.RasterArray(
        data=feature_data,
        transform=raster_template.transform,
        crs=raster_template.crs,
        no_data_value=np.nan,
    )
    return feature_raster
