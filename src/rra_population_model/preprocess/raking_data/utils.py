import geopandas as gpd
import pandas as pd

from rra_population_model.data import PopulationModelData

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


def load_populations(
    pm_data: PopulationModelData, gbd_version: str, wpp_version: str
) -> dict[str, pd.DataFrame]:
    populations = {
        "gbd": pm_data.load_gbd_raking_input("population", f"gbd_{gbd_version}"),
        "fhs": pm_data.load_gbd_raking_input("population", f"fhs_{gbd_version}"),
        "wpp": load_wpp_populations(pm_data, wpp_version),
    }
    return populations


def load_hierarchies(
    pm_data: PopulationModelData, gbd_version: str
) -> dict[str, pd.DataFrame]:
    hierarchies = {
        "gbd": pm_data.load_gbd_raking_input("hierarchy", f"gbd_{gbd_version}"),
        "fhs": pm_data.load_gbd_raking_input("hierarchy", f"fhs_{gbd_version}"),
        "lsae": pm_data.load_gbd_raking_input("hierarchy", "lsae"),
    }
    return hierarchies


def load_shapes(
    pm_data: PopulationModelData, gbd_version: str
) -> dict[str, gpd.GeoDataFrame]:
    shapes = {
        "gbd": pm_data.load_gbd_raking_input("shapes", f"gbd_{gbd_version}"),
        "lsae": pm_data.load_gbd_raking_input("shapes", "lsae_a0"),
    }
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
    wpp: pd.DataFrame, gbd_hierarchy: pd.DataFrame, lsae_hierarchy: pd.DataFrame
) -> pd.DataFrame:
    keep_cols = ["location_id", "location_name", "region_id", "ihme_loc_id"]
    a0_level = 3
    country_mask = gbd_hierarchy["level"] == a0_level
    gbd_metadata = gbd_hierarchy.loc[country_mask, keep_cols].copy()

    # This is a manual mapping of locations that are not linked in the IHME mapping file
    # from the GBD location id to the GBD region in which they reside.
    # We'll grab the additional iso3 and location name metadata from the LSAE location hierarchy
    # since they are not present in the GBD hierarchy. We have to manually include the region_id
    # because the LSAE hierarchy does not consider regions in their hierarchy.
    wpp_missing_metadata = [
        # Tuples of (location_id, region_id)
        # From the IHME mapping file /home/j/DATA/IHME_COUNTRY_CODES/IHME_COUNTRY_CODES_Y2013M07D26.CSV
        (299, 104),  # Anguilla
        (300, 104),  # Aruba
        (313, 104),  # Cayman Islands
        (331, 96),  # Falkland Islands (Islas Malvinas)
        (332, 73),  # Faroe Islands
        (338, 104),  # French Guiana
        (339, 21),  # French Polynesia
        (345, 73),  # Gibraltar
        (350, 104),  # Guadeloupe
        (352, 73),  # Guernsey
        (353, 73),  # Holy See (Vatican City)
        (355, 73),  # Isle of Man
        (356, 73),  # Jersey
        (360, 73),  # Liechtenstein
        (363, 104),  # Martinique
        (368, 104),  # Montserrat
        (372, 21),  # New Caledonia
        (387, 9),  # Reunion
        (391, 104),  # Saint Barthelemy
        (394, 104),  # Saint Martin
        (395, 100),  # Saint Pierre and Miquelon
        (415, 104),  # Turks and Caicos Islands
        (421, 104),  # British Virgin Islands
        (423, 21),  # Wallis and Futuna Islands
        (424, 138),  # Western Sahara
        # Countries not linked in the IHME mapping file
        (60922, 174),  # Bonaire, Sint Eustatius and Saba
        (4641, 104),  # Curacao
        (364, 174),  # Mayotte
        (4642, 174),  # Sint Maarten (Dutch part)
        # Replaced from the IHME mapping file
        (
            60927,
            199,
        ),  # Saint Helena, Ascension, and Tristan da Cunha, replaced (392, 199) with LSAE def
    ]
    # The LSAE hierarchy is missing some iso3 codes, so we'll manually add them here
    supplemental_iso_codes = {
        60922: "BES",
        4641: "CUW",
        4642: "SXM",
        60927: "SHN",
    }
    missing_df = make_location_region_mapping(
        missing_region_metadata=wpp_missing_metadata,
        reference_hierarchy=lsae_hierarchy,
        supplemental_codes=supplemental_iso_codes,
    )

    ihme_metadata = pd.concat([gbd_metadata, missing_df], axis=0, ignore_index=True)

    wpp = wpp.merge(
        ihme_metadata, left_on="iso3", right_on="ihme_loc_id", how="left"
    ).drop(columns="iso3")

    wpp["location_id"] = wpp["location_id"].astype(int)
    wpp["region_id"] = wpp["region_id"].astype(int)

    # Check for missing metadata
    for column in ["location_id", "region_id", "location_name", "ihme_loc_id"]:
        if wpp[column].isna().any():
            msg = f"Missing metadata for {column} in WPP data"
            raise ValueError(msg)

    return wpp


def supplement_wpp(wpp: pd.DataFrame, lsae_hierarchy: pd.DataFrame) -> pd.DataFrame:
    # These locations are not present in either WPP or GBD, but have
    # admin0 level data in the LSAE hierarchy. Some of them are entirely unmodeled
    # and not included in regional aggregates (either intentionally or by mistake)
    # while others just have no population and so don't appear in either hierarchy.

    unmodeled_locations = [
        # Tuples of (location_id, region_id)
        # These are in the IHME mapping but have no WPP estimates
        (297, 73),  # Aland Islands
        (318, 9),  # Christmas Island
        (319, 9),  # Cocos (Keeling) Islands
        (375, 21),  # Norfolk Island
        (382, 21),  # Pitcairn Islands
        (411, 73),  # Svalbard and Jan Mayen
        # These are not present in the IHME mapping and have no WPP estimates
        (296, 73),  # Akrotiri and Dhekelia
        (53483, 73),  # Northern Cyprus
        (311, 73),  # Canary Islands
    ]
    supplemental_iso_codes = {
        60348: "ATF",
        60924: "HMD",
        60928: "UMI",
        60931: "ZZZ",
        60930: "ZZZ",
        93924: "ZZZ",
        94026: "ZZZ",
        94027: "ZZZ",
    }
    unmodeled_df = make_location_region_mapping(
        missing_region_metadata=unmodeled_locations,
        reference_hierarchy=lsae_hierarchy,
        supplemental_codes=supplemental_iso_codes,
    )
    values = pd.DataFrame(
        {
            "year_id": sorted(wpp["year_id"].unique()),
            "population": float("nan"),
        }
    )
    unmodeled_df = unmodeled_df.merge(values, how="cross")

    zero_population_locations = [
        # Tuples of (location_id, region_id)
        (60921, -1),  # Antarctica
        (94026, 70),  # Ashmore and Cartier Islands, Australasia
        (60923, -1),  # Bouvet Island, near antarctica
        (60925, 159),  # British Indian Ocean Territory, South Asia
        (60930, 124),  # Clipperton Island, Central Latin America
        (94027, 70),  # Coral Sea Islands Territory, Australasia
        (60348, 174),  # French Southern Territories, Eastern SSA
        (60924, 70),  # Heard Island and McDonald Islands, Australasia
        (60931, 5),  # Paracel Islands, East Asia
        (
            60926,
            96,
        ),  # South Georgia and the South Sandwich Islands, Southern Latin America
        (93924, 9),  # Spratly Islands, Southeast Asia
        (60928, 100),  # High income North America
    ]
    supplemental_iso_codes = {
        60348: "ATF",
        60924: "HMD",
        60928: "UMI",
    }
    zero_df = make_location_region_mapping(
        missing_region_metadata=zero_population_locations,
        reference_hierarchy=lsae_hierarchy,
        supplemental_codes=supplemental_iso_codes,
    )
    values = pd.DataFrame(
        {
            "year_id": sorted(wpp["year_id"].unique()),
            "population": 0,
        }
    )
    zero_df = zero_df.merge(values, how="cross")

    return pd.concat([wpp, unmodeled_df, zero_df], axis=0, ignore_index=True)
