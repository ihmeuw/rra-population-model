import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
import shapely
import tqdm
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

DENOMINATORS = [
    # "msftv2_density",
    # "msftv3_density",
    "msftv4_density",
    "ghsl_density",
    "ghsl_residential_density",
    "ghsl_volume",
    "ghsl_residential_volume",
]
EXCLUDE_FEATURES = [
    *DENOMINATORS,
    "ghsl_nonresidential_density",
    "ghsl_nonresidential_volume",
]

# These "features" are made by this script and added to the features subdirectory.
# Many are left-hand side variables only available for countries with training data.
NEW_PIXEL_FEATURES = [
    # "urban",
    "population",
    "population_density",
    "admin_is_multi_tile",
] + [f"occupancy_rate_{d}" for d in DENOMINATORS]


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
    pm_data: PopulationModelData,
    model_frame: gpd.GeoDataFrame,
    resolution: str,
    tile_key: str,
    block_key: str,
    time_point: str,
    tile_poly: shapely.Polygon,
    admin_population: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Load the raster features for the tile and convert to a GeoDataFrame."""
    features_to_load = pm_data.list_features(resolution, block_key, time_point)
    tile_features = {}
    for feature_name in features_to_load:
        tile_features[feature_name] = pm_data.load_feature(
            resolution=resolution,
            block_key=block_key,
            feature_name=feature_name,
            time_point=time_point,
            subset_bounds=tile_poly,
        )

    bd_raster = tile_features.pop("msftv2_density")
    feature_gdf = (
        bd_raster.to_gdf()
        .reset_index()
        .rename(columns={"value": "pixel_msftv2_density", "index": "pixel_id"})
        .sort_values("pixel_id")
    )
    feature_gdf["pixel_area"] = feature_gdf.area
    feature_gdf["block_key"] = block_key
    feature_gdf["tile_key"] = tile_key
    feature_gdf["time_point"] = time_point

    for feature_name, feature_raster in tile_features.items():
        feature_gdf[f"pixel_{feature_name}"] = feature_raster.to_numpy().flatten()

    model_tile_gdf = model_frame[model_frame.tile_key == tile_key].copy()
    model_tile_gdf["geometry"] = model_tile_gdf.buffer(10)
    admin_population["admin_id"] = admin_population["shape_id"]
    admin_population["admin_area"] = admin_population.area
    admin_population["geometry"] = admin_population.buffer(0)
    tile_gdf = (
        admin_population.rename(
            columns={
                "population_total": "admin_population",
            }
        )
        # First we just want to subset the admins as the computational complexity
        # of overlay is proportional to the area of the two gdfs (or to the points
        # it has to figure out).
        .overlay(
            model_tile_gdf[["geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        .overlay(
            feature_gdf,
            how="intersection",
            keep_geom_type=True,
        )
    )
    tile_gdf["isection_area"] = tile_gdf.area
    return tile_gdf


def process_model_gdf(
    model_gdf: gpd.GeoDataFrame, features: list[str]
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

    for denominator in DENOMINATORS:
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

    for feature in features:
        model_gdf[f"admin_{feature}"] = (
            model_gdf[f"pixel_{feature}"] * model_gdf["admin_area_weight"]
        )
        model_gdf[f"admin_{feature}"] = model_gdf.groupby("admin_id")[
            f"admin_{feature}"
        ].transform("sum")

    return model_gdf


def filter_to_admin_gdf(
    model_gdf: gpd.GeoDataFrame,
    admins: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter the model GDF to only the admin-level features and rows."""
    keep_cols = (
        ["block_key", "tile_key", "time_point"]
        + [c for c in model_gdf if c[:5] == "admin"]
        + ["geometry"]
    )
    model_gdf = model_gdf.loc[:, keep_cols].groupby("admin_id").first().reset_index()
    model_gdf["geometry"] = (
        admins.set_index("shape_id").loc[model_gdf["admin_id"], "geometry"].to_numpy()
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


def training_data_main(
    resolution: str,
    iso3_list: str,
    tile_key: str,
    time_point: str,
    model_root: str | Path,
) -> None:
    """Build the training data for the MEX model for a single tile."""
    print("Loading metadata")
    year = time_point.split("q")[0]
    pm_data = PopulationModelData(model_root)
    model_frame = pm_data.load_modeling_frame(resolution)
    tile_meta = model_frame[model_frame.tile_key == tile_key]
    block_key = tile_meta.block_key.iloc[0]
    tile_poly = tile_meta.geometry.iloc[0]

    print("Loading and compiling admin census data")
    admin_data = []
    for iso in iso3_list.split(","):
        a = pm_data.load_census_data(iso, year, tile_poly)
        # Need to intersect again with the tile poly because we load based on the
        # intersection with the bounding box.
        a = a[(a.admin_level == a.admin_level.max()) & (a.intersects(tile_poly))]
        admin_data.append(a)

    admins = pd.concat(admin_data, ignore_index=True)

    if admins.empty:
        print(f"No admins found for tile {tile_key}. Likely an all water tile.")
        return

    print("Finding tile neighborhood")
    try:
        full_shape = admins.buffer(0).union_all()
    except shapely.errors.GEOSException:
        # Buffer usually fixes small topological errors, but
        # for at least one tile it causes a GEOS exception.
        # This should be investigated, but just going with something
        # that appears to work for now.
        full_shape = admins.union_all()

    tile_neighborhood = model_frame[
        model_frame.intersects(full_shape)
    ].tile_key.tolist()

    print("Loading model gdfs")
    model_gdfs = []
    for n_tile_key in tile_neighborhood:
        print(n_tile_key)
        n_tile_meta = model_frame[model_frame.tile_key == n_tile_key]
        n_block_key = n_tile_meta.block_key.iloc[0]
        n_tile_poly = n_tile_meta.geometry.iloc[0]
        n_tile_gdf = get_tile_feature_gdf(
            pm_data,
            model_frame,
            resolution,
            n_tile_key,
            n_block_key,
            time_point,
            n_tile_poly,
            admins,
        )
        if not n_tile_gdf.empty:
            model_gdfs.append(n_tile_gdf)

    print("Processing model gdf")
    model_gdf = pd.concat(model_gdfs, ignore_index=True)

    features = [
        f
        for f in pm_data.list_features(resolution, block_key, time_point)
        if f not in EXCLUDE_FEATURES
    ]

    model_gdf = process_model_gdf(model_gdf, features)
    admin_gdf = filter_to_admin_gdf(model_gdf, admins)
    tile_gdf = model_gdf[model_gdf["tile_key"] == tile_key]

    print("Calculating pixel area weights")
    pixel_area_weight = (
        tile_gdf.groupby(["admin_id", "pixel_id"])[["admin_area_weight"]]
        .first()
        .reset_index()
    )

    print("rasterizing features")
    raster_template = pm_data.load_feature(
        resolution, block_key, "msftv2_density", time_point, tile_poly
    )

    training_rasters = [
        f"{measure}_{denominator}"
        for measure, denominator in itertools.product(
            ["population", "occupancy_rate", "log_occupancy_rate"], DENOMINATORS
        )
    ]
    training_rasters.append("multi_tile")
    tile_rasters = {}
    for raster_name in training_rasters:
        raster = raster_from_pixel_feature(tile_gdf, raster_name, raster_template)
        tile_rasters[raster_name] = raster

    print("Saving")
    pm_data.save_tile_training_data(
        tile_key,
        admin_gdf,
        pixel_area_weight,
        tile_rasters,
    )


@click.command()  # type: ignore[arg-type]
@click.option("--iso3-list", type=str, required=True)
@clio.with_resolution()
@clio.with_tile_key()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
def training_data_task(
    resolution: str,
    iso3_list: str,
    tile_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    """Build the response for a given tile and time point."""
    training_data_main(
        resolution,
        iso3_list,
        tile_key,
        time_point,
        output_dir,
    )


def get_training_locations_and_years(
    pm_data: PopulationModelData,
) -> list[tuple[str, str, str]]:
    """Get the locations and years for which we have training data."""
    available_census_years = pm_data.list_census_data()  # noqa: F841
    return [
        ("MEX", "2020", "1"),
        ("USA", "2020", "1"),
    ]


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def training_data(
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Build the training data for the MEX model."""
    pm_data = PopulationModelData(output_dir)

    print("Loading the modeling frame.")
    modeling_frame = pm_data.load_modeling_frame(resolution)
    training_census_years = get_training_locations_and_years(pm_data)

    buffer_size = 5000
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
    print(f"Building test/train data for {len(to_run)} tiles.")

    status = jobmon.run_parallel(
        task_name="training_data",
        flat_node_args=(("tile-key", "time-point", "iso3-list"), to_run),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        runner="pmtask preprocess",
        max_attempts=5,
    )

    if status != "D":
        msg = f"Workflow failed with status {status}."
        raise RuntimeError(msg)

    print("Loading the model GDFs.")
    model_gdfs = []
    for tile_dir in tqdm.tqdm(list(pm_data.tile_training_data.iterdir())):
        if "zzz" in tile_dir.name or tile_dir.is_file():
            continue
        tile_pps = pm_data.load_people_per_structure(tile_dir.name)
        tile_paw = pm_data.load_pixel_area_weights(tile_dir.name)
        model_gdfs.append((tile_pps, tile_paw))

    pps, paw = zip(*model_gdfs, strict=False)

    people_per_structure_gdf = pd.concat(pps, ignore_index=True)
    pixel_area_weight_gdf = pd.concat(paw, ignore_index=True)
    pm_data.save_summary_training_data(
        people_per_structure_gdf,
        pixel_area_weight_gdf,
    )
