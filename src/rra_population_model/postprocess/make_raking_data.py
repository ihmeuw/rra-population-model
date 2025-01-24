from pathlib import Path

import geopandas as gpd
import pandas as pd
from db_queries import (  # type: ignore[import-not-found]
    get_location_metadata,
    get_population,
)

RELEASE_ID = 16
LOCATION_SET_ID = 22
OUT_ROOT = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/people_per_structure/raking_data"
)


def make_gbd_hierarchy_and_pop() -> None:
    h = get_location_metadata(location_set_id=LOCATION_SET_ID, release_id=RELEASE_ID)
    h.to_parquet(
        "/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/gbd_hierarchy.parquet"
    )

    location_ids = h.location_id.tolist()
    pop = get_population(
        location_id=location_ids,
        year_id=list(range(1950, 2023)),
        age_group_id=22,
        sex_id=3,
        release_id=9,
    )
    pop = pop[["location_id", "year_id", "population"]]
    pop = h[
        ["location_id", "parent_id", "location_name", "most_detailed", "level"]
    ].merge(pop, on="location_id")

    bcp = [44790, 44783]
    new_bcp_rows = (
        pop[pop.location_id.isin(bcp)]
        .groupby(["parent_id", "year_id"])
        .population.sum()
        .reset_index()
    )
    new_bcp_rows["location_id"] = 60909
    new_bcp_rows["location_name"] = "Bournemouth, Christchurch, and Poole"
    new_bcp_rows["most_detailed"] = 1
    new_bcp_rows["level"] = 6

    pop = pd.concat([pop[~pop.location_id.isin(bcp)], new_bcp_rows], ignore_index=True)

    dorset = pop.location_id == 44779  # noqa: PLR2004
    pop.loc[dorset, "location_id"] = 60910
    pop.loc[dorset, "location_name"] = "Dorset Council"
    pop.to_parquet(
        "/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/gbd_population.parquet"
    )


def make_gbd_shapes() -> None:
    gbd = gpd.read_file(
        "/home/j/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2022/master/shapefiles/GBD2022_analysis_final.shp"
    )

    col_map = {
        "loc_id": "location_id",
        "parent_id": "parent_id",
        "loc_name": "location_name",
        "geometry": "geometry",
    }

    gbd = gbd.rename(columns=col_map).loc[:, list(col_map.values())]

    drop = [
        44533,  # China without HK and Macao, should not be in this shapefile.
    ]
    gbd = gbd[~gbd.location_id.isin(drop)]

    # Ethiopia snnp should be unified
    snnp = [95069, 60908, 94364]
    new_snnp_shp = (
        gbd[gbd.location_id.isin(snnp)].dissolve(by="parent_id").geometry.iloc[0]
    )
    snnp_row = gpd.GeoDataFrame(
        {
            "location_id": [44858],
            "parent_id": [179],
            "location_name": ["Southern Nations, Nationalities, and Peoples"],
            "geometry": [new_snnp_shp],
        },
        crs=gbd.crs,
        index=[0],
    )

    # Old split in the UK
    nhs = [60911, 60912]
    new_nhs_shp = (
        gbd[gbd.location_id.isin(nhs)].dissolve(by="parent_id").geometry.iloc[0]
    )
    nhs_row = gpd.GeoDataFrame(
        {
            "location_id": [44693],
            "parent_id": [4621],
            "location_name": ["Northamptonshire"],
            "geometry": [new_nhs_shp],
        },
        crs=gbd.crs,
        index=[0],
    )

    gbd = pd.concat(
        [
            gbd[~gbd.location_id.isin(snnp + nhs)],
            snnp_row,
            nhs_row,
        ],
        ignore_index=True,
    )
    gbd.to_parquet(
        "/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/gbd_shapes.parquet"
    )
