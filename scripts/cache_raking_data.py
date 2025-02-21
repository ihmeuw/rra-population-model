"""This script caches GBD data used to making raking inputs.

Retrieving GBD data requires access to the GBD shared functions, which
pin some dependencies for specific things like numpy and pandas. This
is incompatible with our modeling environment, so we need to cache the
data in a way that we can use it in our modeling environment.

This script needs to be run from a cluster node with access to the J-drive.
"""

from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd
import xarray as xr
from db_queries import (  # type: ignore[import-not-found]
    get_location_metadata,
    get_population,
)
from rra_tools.shell_tools import mkdir, touch

GBD_RELEASE_ID = 9
GBD2023_RELEASE_ID = 16
FHS_RELEASE_ID = 9
GBD_LOCATION_SET_ID = 22
FHS_LOCATION_SET_ID = 39
LSAE_LOCATION_SET_ID = 125


MODEL_ROOT = Path("/mnt/team/rapidresponse/pub/population-model")


def load_gbd_populations(location_set_id: int, release_id: int) -> pd.DataFrame:
    return (  # type: ignore[no-any-return]
        get_population(
            release_id=release_id,
            location_id="all",
            location_set_id=location_set_id,
            year_id="all",
        )
        .drop(columns=["age_group_id", "sex_id", "run_id"])
        .set_index(["location_id", "year_id"])
        .sort_index()
        .reset_index()
    )


def load_fhs_population(*args: Any, **kwargs: Any) -> pd.DataFrame:
    pop_fhs_path = "/mnt/share/forecasting/data/9/future/population/20250219_draining_fix_old_pop_v5/summary/summary.nc"
    return (
        xr.open_dataset(pop_fhs_path)
        .sel(scenario=0, statistic="mean", sex_id=3, age_group_id=22)
        .to_dataframe()
        .reset_index()
        .drop(columns=["scenario", "sex_id", "age_group_id", "statistic"])
        .set_index(["location_id", "year_id"])
        .sort_index()
        .reset_index()
        .rename(columns={"value": "population"})
    )


def load_gbd_shapes(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    column_map = {
        "loc_id": "location_id",
        "parent_id": "parent_id",
        "level": "level",
        "loc_name": "location_name",
        "spr_reg_id": "super_region_id",
        "region_id": "region_id",
        "ihme_lc_id": "ihme_loc_id",
        "geometry": "geometry",
    }
    gdf = gdf.rename(columns=column_map).loc[:, column_map.values()]

    return gdf


def load_lsae_shapes(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    admin_level = path.stem[-1]
    column_map = {
        "loc_id": "location_id",
        "geometry": "geometry",
        f"ADM{admin_level}_NAME": "location_name",
    }
    gdf = gdf.rename(columns=column_map).loc[:, column_map.values()]

    return gdf


@click.command()
@click.option("--model-root", type=click.Path(exists=True), default=MODEL_ROOT)
def cache_raking_data(model_root: str) -> None:
    data_out_root = Path(model_root) / "admin-inputs" / "raking" / "gbd-inputs"
    mkdir(data_out_root, exist_ok=True, parents=True)

    # Cache WPP data
    wpp_paths = {
        "2022": Path(
            "/home/j/DATA/UN_WORLD_POPULATION_PROSPECTS/2022/UN_WPP_2022_GEN_F01_DEMOGRAPHIC_INDICATORS_1950_2100_Y2022M07D11.XLSX"
        ),
        "2024": Path(
            "/home/j/DATA/UN_WORLD_POPULATION_PROSPECTS/2024/UN_WPP_2024_GEN_F01_DEMOGRAPHIC_INDICATORS_REV1_Y2024M07D31.XLSX"
        ),
    }
    for year, path in wpp_paths.items():
        print(f"Caching WPP data for {year}")
        estimates = pd.read_excel(path, header=16, sheet_name="Estimates")
        forecasts = pd.read_excel(path, header=16, sheet_name="Medium variant")
        wpp = pd.concat([estimates, forecasts], axis=0, ignore_index=True)
        col_map = {
            "ISO3 Alpha-code": "iso3",
            "Total Population, as of 1 July (thousands)": "population",
            "Year": "year_id",
            "Type": "type",
        }
        wpp = wpp.rename(columns=col_map).loc[:, col_map.values()]  # type: ignore[index]
        wpp = (
            wpp.loc[wpp["type"] == "Country/Area", :]
            .reset_index(drop=True)
            .drop(columns="type")
        )
        wpp["year_id"] = wpp["year_id"].astype(int)
        wpp["population"] = wpp["population"] * 1000

        out_path = data_out_root / f"population_wpp_{year}.parquet"
        touch(out_path, clobber=True)
        wpp.to_parquet(out_path)

    hierarchy_specs = {
        "gbd_2021": (GBD_LOCATION_SET_ID, GBD_RELEASE_ID),
        "gbd_2023": (GBD_LOCATION_SET_ID, GBD2023_RELEASE_ID),
        "fhs_2021": (FHS_LOCATION_SET_ID, FHS_RELEASE_ID),
        "lsae": (LSAE_LOCATION_SET_ID, GBD2023_RELEASE_ID),
    }
    for name, (location_set_id, release_id) in hierarchy_specs.items():
        print(f"Caching hierarchy for {name}")
        h = get_location_metadata(
            location_set_id=location_set_id, release_id=release_id
        )
        out_path = data_out_root / f"hierarchy_{name}.parquet"
        touch(out_path, clobber=True)
        h.to_parquet(out_path)

    pop_loaders = {
        "gbd_2021": (load_gbd_populations, (GBD_LOCATION_SET_ID, GBD_RELEASE_ID)),
        "gbd_2023": (load_gbd_populations, (GBD_LOCATION_SET_ID, GBD2023_RELEASE_ID)),
        "fhs_2021": (load_fhs_population, ()),
    }
    for name, (loader, args) in pop_loaders.items():
        print(f"Caching population for {name}")
        pop = loader(*args)  # type: ignore[operator]
        out_path = data_out_root / f"population_{name}.parquet"
        touch(out_path, clobber=True)
        pop.to_parquet(out_path)

    gbd_2021_shape_path = Path(
        "/home/j/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2021/master/shapefiles/GBD2021_analysis_final_loc_set_22.shp"
    )
    gbd_2023_shape_path = Path(
        "/home/j/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2023/master/shapefiles/GBD2023_analysis_final_loc_set_22.shp"
    )
    lsae_shape_root = Path("/home/j/WORK/11_geospatial/admin_shapefiles/current/")
    shape_loaders = {
        "gbd_2021": (load_gbd_shapes, gbd_2021_shape_path),
        "gbd_2023": (load_gbd_shapes, gbd_2023_shape_path),
        "lsae_a0": (load_lsae_shapes, lsae_shape_root / "lbd_standard_admin_0.shp"),
        "lsae_a1": (load_lsae_shapes, lsae_shape_root / "lbd_standard_admin_1.shp"),
        "lsae_a2": (load_lsae_shapes, lsae_shape_root / "lbd_standard_admin_2.shp"),
    }
    for name, (loader, path) in shape_loaders.items():
        print(f"Caching shape for {name}")
        shape = loader(path)
        out_path = data_out_root / f"shapes_{name}.parquet"
        touch(out_path, clobber=True)
        shape.to_parquet(out_path, write_covering_bbox=True)


if __name__ == "__main__":
    cache_raking_data()
