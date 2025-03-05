from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from rra_population_model.data import PopulationModelData
from rra_population_model.preprocess.raking_data.metadata import (
    NO_REGION_ID,
    SUPPLEMENT,
)

###########
# Loaders #
###########


def load_wpp_populations(
    pm_data: PopulationModelData, wpp_version: str
) -> pd.DataFrame:
    if wpp_version == "2022":
        wpp = pm_data.load_gbd_raking_input("population", f"wpp_{wpp_version}")
        # Merge Hong Kong, Macau, and Kosovo into China and Serbia, respectively
        merge_map = {
            # GBD treats Hong Kong and Macau as part of China at the admin0 level
            "CHN": ["CHN", "HKG", "MAC"],
            # GBD treats Kosovo as part of Serbia at the admin0 level
            "SRB": ["SRB", "XKX"],
        }
        for target, sources in merge_map.items():
            mask = wpp["iso3"].isin(sources)
            merged = (
                wpp.loc[mask, :]
                .groupby("year_id")
                .sum()
                .reset_index()
                .assign(iso3=target)
            )
            wpp = (
                pd.concat([wpp.loc[~mask, :], merged])
                .sort_values(["iso3", "year_id"])
                .reset_index(drop=True)
            )
    elif wpp_version == "2024":
        msg = "We still need to implement/validate the merge_map for WPP 2024"
        raise NotImplementedError(msg)
    else:
        msg = f"Invalid WPP version: {wpp_version}"
        raise ValueError(msg)
    return wpp


def load_ihme_populations(
    pm_data: PopulationModelData,
    gbd_version: str,
) -> dict[str, pd.DataFrame]:
    populations = {
        "gbd": pm_data.load_gbd_raking_input("population", f"gbd_{gbd_version}"),
        "fhs": pm_data.load_gbd_raking_input("population", f"fhs_{gbd_version}"),
    }
    return populations


def load_hierarchies(
    pm_data: PopulationModelData, gbd_version: str
) -> dict[str, pd.DataFrame]:
    hierarchies = {
        "gbd": pm_data.load_gbd_raking_input("hierarchy", f"gbd_{gbd_version}"),
        "fhs": pm_data.load_gbd_raking_input("hierarchy", f"fhs_{gbd_version}"),
    }
    keep_cols = [
        "location_id",
        "location_name",
        "ihme_loc_id",
        "parent_id",
        "region_id",
        "level",
        "most_detailed",
    ]
    hierarchies = {k: v.loc[:, keep_cols] for k, v in hierarchies.items()}
    return hierarchies


def load_shapes(
    pm_data: PopulationModelData, gbd_version: str
) -> dict[str, gpd.GeoDataFrame]:
    shapes = {
        "gbd": pm_data.load_gbd_raking_input("shapes", f"gbd_{gbd_version}"),
        "lsae": pm_data.load_gbd_raking_input("shapes", "lsae_1285_a0"),
    }
    keep_cols = ["location_id", "geometry"]
    shapes = {k: v.loc[:, keep_cols] for k, v in shapes.items()}
    return shapes


###########
# Helpers #
###########


def make_location_region_mapping(
    missing_region_metadata: list[tuple[int, int]],
    reference_hierarchy: pd.DataFrame,
    supplemental_codes: dict[int, str],
) -> pd.DataFrame:
    """Make a location-region mapping DataFrame.

    Parameters
    ----------
    missing_region_metadata
        A list of tuples where each tuple is a location_id and a region_id.
    reference_hierarchy
        The canonical location metadata to map from.
    supplemental_codes
        A dictionary mapping location_ids to ISO3 codes where those iso3
        codes are not present in the reference hierarchy.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns location_id, region_id, location_name, and ihme_loc_id.

    """
    # Subset the hierarchy to just the location metadata we need
    h = reference_hierarchy.loc[
        :, ["location_id", "location_name", "ihme_loc_id"]
    ].copy()
    # Build the mapping for the missing region metadata
    mapping = pd.DataFrame(
        missing_region_metadata, columns=["location_id", "region_id"]
    )
    mapping = mapping.merge(h, on="location_id", how="left")

    # Add in any supplemental codes we're missing.
    for location_id, iso3 in supplemental_codes.items():
        mapping.loc[mapping["location_id"] == location_id, "ihme_loc_id"] = iso3
    return mapping


def add_gbd_metadata_to_wpp(
    wpp: pd.DataFrame,
    gbd_hierarchy: pd.DataFrame,
    supplemental_metadata: pd.DataFrame,
) -> pd.DataFrame:
    keep_cols = ["location_id", "location_name", "region_id", "ihme_loc_id"]
    a0_level = 3
    country_mask = gbd_hierarchy["level"] == a0_level
    gbd_metadata = gbd_hierarchy.loc[country_mask, keep_cols].copy()
    gbd_metadata["region_id"] = gbd_metadata["region_id"].astype(int)

    wpp_supplement = supplemental_metadata.loc[
        supplemental_metadata.category == SUPPLEMENT.WPP
    ].drop(columns=["category"])

    ihme_metadata = pd.concat(
        [
            gbd_metadata,
            wpp_supplement,
        ],
        ignore_index=True,
    )

    wpp = (
        wpp.merge(ihme_metadata, left_on="iso3", right_on="ihme_loc_id", how="left")
        .drop(columns="iso3")
        .sort_values(["location_id", "year_id"])
    )
    return wpp


def add_unmodeled_and_zero_population_locations(
    wpp: pd.DataFrame,
    supplemental_metadata: pd.DataFrame,
) -> pd.DataFrame:
    data = [wpp]

    value_map = {
        SUPPLEMENT.UNMODELED: np.nan,
        SUPPLEMENT.ZERO_POPULATION: 0,
    }

    for category, value in value_map.items():
        meta = supplemental_metadata.loc[
            supplemental_metadata["category"] == category
        ].drop(columns=["category"])
        year_values = pd.DataFrame(
            {
                "year_id": sorted(wpp["year_id"].unique()),
                "population": value,
            }
        )
        data.append(meta.merge(year_values, how="cross"))

    return pd.concat(data, ignore_index=True).sort_values(["location_id", "year_id"])


def compute_regional_scalar(
    wpp: pd.DataFrame,
) -> pd.Series[float]:
    region_pop = wpp.groupby(["region_id", "year_id"])["population"].transform("sum")
    scalar = wpp["population"] / region_pop
    scalar[wpp["population"] == 0] = 0.0
    return scalar


def prepare_ihme_population(
    populations: dict[str, pd.DataFrame],
    hierarchies: dict[str, gpd.GeoDataFrame],
    version_tag: str,
) -> pd.DataFrame:
    p_gbd = populations["gbd"].merge(hierarchies["gbd"], on="location_id")

    if version_tag == "fhs":
        p_fhs = populations["fhs"].merge(hierarchies["fhs"], on="location_id")
        keep_mask = (
            # Drop years in GBD that are present in FHS
            ~p_gbd.year_id.isin(p_fhs.year_id.unique())
            # Drop GBD subnationals that are not in FHS so we don't get artifacts in raking
            & p_gbd.location_id.isin(p_fhs.location_id.unique())
        )
        p_gbd = p_gbd.loc[keep_mask]
        # Update most-detailed metadata to match FHS
        fhs_most_detailed = p_gbd.location_id.isin(
            p_fhs.loc[p_fhs.most_detailed == 1].location_id.unique()
        )
        p_gbd.loc[fhs_most_detailed, "most_detailed"] = 1
        assert (p_gbd.loc[~fhs_most_detailed].most_detailed == 0).all()  # noqa: S101
        pop = pd.concat([p_gbd, p_fhs], ignore_index=True)
    elif version_tag == "gbd":
        pop = p_gbd
    else:
        msg = f"Unsupported version tag: {version_tag}"
        raise ValueError(msg)
    pop["region_id"] = pop["region_id"].fillna(NO_REGION_ID).astype(int)
    return pop.sort_values(["location_id", "year_id"])


def compute_missing_populations(
    wpp: pd.DataFrame,
    ihme: pd.DataFrame,
) -> pd.DataFrame:
    # Subset to the wpp locations and years we're
    # using to fill in the missing populations.
    keep_mask = ~wpp.location_id.isin(ihme.location_id.unique()) & (
        wpp.year_id.isin(ihme.year_id.unique())
    )
    wpp_subset = (
        wpp.loc[keep_mask]
        .set_index(["region_id", "year_id"])
        .drop(columns=["population"])
    )

    # Get the remaining population to distribute by grabbing the regional populations
    # which are inclusive of unmodeled locations and subtracting the sum of the most
    # detailed locations within each region.
    region_pop = ihme.loc[ihme.location_id == ihme.region_id].set_index(
        ["region_id", "year_id"]
    )["population"]
    region_agg = (
        ihme.loc[ihme.most_detailed == 1]
        .groupby(["region_id", "year_id"])["population"]
        .sum()
    )
    # Clip to zero to avoid negative populations due to rounding errors.
    pop_to_distribute = (region_pop - region_agg).clip(lower=0.0)

    no_region_pop = pd.DataFrame(
        {"region_id": -1, "year_id": ihme["year_id"].unique(), "population": 0}
    ).set_index(["region_id", "year_id"])
    pop_to_distribute = pd.concat([pop_to_distribute, no_region_pop])["population"]

    # Get a scalar for the remaining population to distribute from the scalar for the
    # entire regional population by renormalizing in our subset.
    adjusted_scalar = wpp_subset["scalar"].divide(
        wpp_subset.groupby(["region_id", "year_id"])["scalar"].transform("sum")
    )
    adjusted_scalar[wpp_subset["scalar"] == 0] = 0.0

    scaled_population = pop_to_distribute.loc[adjusted_scalar.index].multiply(
        adjusted_scalar
    )

    missing_populations = (
        wpp_subset.drop(columns=["scalar"])
        .assign(
            population=scaled_population.values,
            most_detailed=1,
            level=3,
        )
        .reset_index()
        .drop(columns=["region_id"])
        .sort_values(["location_id", "year_id"])
    )

    return missing_populations


def build_raking_population(
    ihme_population: pd.DataFrame,
    wpp_population: pd.DataFrame,
    missing_population: pd.DataFrame,
) -> pd.DataFrame:
    ihme_population = ihme_population[ihme_population.most_detailed == 1]

    full_population = (
        pd.concat([ihme_population, missing_population], ignore_index=True)
        .sort_values(["location_id", "year_id"])
        .reset_index(drop=True)
    )

    # Add on a column with WPP population for UN product raking
    full_population = full_population.merge(
        wpp_population[["location_id", "year_id", "population"]].rename(
            columns={"population": "wpp_population"}
        ),
        on=["location_id", "year_id"],
        how="left",
    )

    return full_population


def build_raking_shapes(
    shapes: dict[str, gpd.GeoDataFrame],
    raking_population: pd.DataFrame,
) -> gpd.GeoDataFrame:
    ihme_shapes = shapes["gbd"]
    keep_mask = ihme_shapes["location_id"].isin(raking_population["location_id"])
    ihme_shapes = ihme_shapes.loc[keep_mask]

    lsae_shapes = shapes["lsae"]
    keep_mask = lsae_shapes["location_id"].isin(
        raking_population["location_id"]
    ) & ~lsae_shapes["location_id"].isin(ihme_shapes["location_id"])
    missing_shapes = lsae_shapes.loc[keep_mask]

    full_shapes = pd.concat(
        [ihme_shapes, missing_shapes], ignore_index=True
    ).sort_values("location_id")

    return full_shapes


def validate_raking_data(
    raking_population: pd.DataFrame,
    raking_shapes: gpd.GeoDataFrame,
) -> None:
    extra = set(raking_shapes.location_id) - set(raking_population.location_id)
    missing = set(raking_population.location_id) - set(raking_shapes.location_id)
    if extra:
        msg = f"Extra location_ids in shapes: {sorted(extra)}"
        raise ValueError(msg)
    if missing:
        msg = f"Missing location_ids in shapes: {sorted(missing)}"
        raise ValueError(msg)
