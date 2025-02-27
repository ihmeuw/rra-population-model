import click
import pandas as pd

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.preprocess.raking_data import (
    utils,
)

HIERARCHY_KEEP_COLS = [
    "parent_id",
    "location_id",
    "most_detailed",
    "level",
    "location_name",
    "ihme_loc_id",
]


def raking_data_main(  # noqa: PLR0915
    output_dir: str,
    out_version: str,
    wpp_version: str = "2022",
) -> None:
    pm_data = PopulationModelData(output_dir)

    version_tag, gbd_version = out_version.split("_")

    print("Loading data...")
    hierarchies = utils.load_hierarchies(pm_data, gbd_version=gbd_version)
    populations = utils.load_populations(
        pm_data, gbd_version=gbd_version, wpp_version=wpp_version
    )
    shapes = utils.load_shapes(pm_data, gbd_version=gbd_version)

    print("Building WPP data...")
    # Add GBD location and region ids to the WPP data by mapping on iso3 codes
    wpp = utils.add_gbd_metadata_to_wpp(
        wpp=populations["wpp"],
        gbd_hierarchy=hierarchies["gbd"],
        lsae_hierarchy=hierarchies["lsae"],
    )
    # Supplement missing or zero population locations with nans or zeroes respectively
    wpp = utils.supplement_wpp(
        wpp=wpp,
        lsae_hierarchy=hierarchies["lsae"],
    )

    # Compute the population scalar: the location fraction of the regional population
    wpp["region_pop"] = wpp.groupby(["region_id", "year_id"])["population"].transform(
        "sum"
    )
    wpp["scalar"] = wpp["population"] / wpp["region_pop"]
    wpp.loc[wpp.population == 0, "scalar"] = 0.0

    h_gbd = hierarchies["gbd"].loc[:, HIERARCHY_KEEP_COLS]
    p_gbd = populations["gbd"].merge(h_gbd, on="location_id")
    p_gbd = p_gbd.loc[
        p_gbd.year_id <= int(gbd_version), :
    ]  # Drop the terminal year of the GBD estimates, these are produced by fhs

    # Drop GBD subnationals as they're not present in fhs and will introduce artifacts
    if version_tag == "fhs":
        h_fhs = hierarchies["fhs"].loc[:, HIERARCHY_KEEP_COLS]
        p_fhs = populations["fhs"].merge(h_fhs, on="location_id")
        p_gbd = p_gbd[p_gbd.location_id.isin(p_fhs.location_id.unique())]
        most_detailed_locs = p_fhs[p_fhs.most_detailed == 1].location_id.unique()
        p_gbd.loc[p_gbd.location_id.isin(most_detailed_locs), "most_detailed"] = 1
        p_gbd.loc[~p_gbd.location_id.isin(most_detailed_locs), "most_detailed"] = 0
        p = pd.concat([p_gbd, p_fhs], axis=0, ignore_index=True)
    else:
        p = p_gbd
        wpp = wpp.loc[wpp.year_id <= int(gbd_version), :]

    p = p.sort_values(["location_id", "year_id"])

    wpp_subset = wpp.loc[
        ~wpp.location_id.isin(p.location_id.unique()),
        ["location_id", "year_id", "region_id", "location_name", "scalar"],
    ]

    gbd_region_ids = list(set(wpp_subset["region_id"]) - {-1})
    gbd_region_pops = (
        p.set_index(["location_id", "year_id"])["population"]
        .loc[gbd_region_ids]
        .reset_index()
    )
    no_region_pop = pd.DataFrame(
        {"location_id": -1, "year_id": wpp_subset["year_id"].unique(), "population": 0}
    )
    region_pops = pd.concat(
        [gbd_region_pops, no_region_pop], axis=0, ignore_index=True
    ).set_index(["location_id", "year_id"])["population"]

    wpp_subset["gbd_region_pop"] = region_pops.loc[
        wpp_subset.set_index(["region_id", "year_id"]).index
    ].to_numpy()
    wpp_subset["population"] = wpp_subset["gbd_region_pop"] * wpp_subset["scalar"]
    wpp_subset["parent_id"] = wpp_subset["region_id"]
    wpp_subset = wpp_subset.loc[
        :, ["parent_id", "location_id", "year_id", "population", "location_name"]
    ].assign(most_detailed=1, level=3)

    # Drop GBD aggregates
    p = p.loc[~p.level.isin([0, 1, 2])]

    # Add the supplemental data to the population data
    full_pop = pd.concat([p, wpp_subset], axis=0, ignore_index=True)
    # Also add a column for wpp population
    full_pop = full_pop.merge(
        wpp[["location_id", "year_id", "population"]].rename(
            columns={"population": "wpp_population"}
        ),
        on=["location_id", "year_id"],
        how="outer",
    )

    gbd_shapes = shapes["gbd"]
    gbd_shapes = gbd_shapes.loc[
        gbd_shapes.location_id.isin(p.location_id.unique()), ["location_id", "geometry"]
    ]
    lsae_shapes = shapes["lsae"]
    supplement_shapes = lsae_shapes.loc[
        lsae_shapes.location_id.isin(wpp_subset.location_id.unique()),
        ["location_id", "geometry"],
    ]
    full_shapes = pd.concat([gbd_shapes, supplement_shapes], axis=0, ignore_index=True)

    extra = set(full_shapes.location_id) - set(full_pop.location_id)
    missing = set(full_pop.location_id) - set(full_shapes.location_id)
    if extra:
        msg = f"Extra location_ids in shapes: {extra}"
        raise ValueError(msg)
    if missing:
        msg = f"Missing location_ids in shapes: {missing}"
        raise ValueError(msg)

    pm_data.save_raking_data(full_pop, full_shapes, f"{out_version}_wpp_{wpp_version}")


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_choice("out_version", allow_all=False, choices=["gbd_2021", "fhs_2021"])
def raking_data(
    output_dir: str,
    out_version: str,
) -> None:
    """Prepare model features."""
    raking_data_main(output_dir, out_version)
