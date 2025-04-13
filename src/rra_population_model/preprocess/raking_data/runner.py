import click

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.preprocess.raking_data import (
    utils,
)
from rra_population_model.preprocess.raking_data.metadata import (
    load_supplmental_metadata,
)


def raking_data_main(
    output_dir: str,
    out_version: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    version_tag, gbd_version = out_version.split("_")

    print("Loading data...")
    # IHME
    hierarchies = utils.load_hierarchies(pm_data, gbd_version=gbd_version)
    populations = utils.load_ihme_populations(pm_data, gbd_version=gbd_version)
    shapes = utils.load_shapes(pm_data, gbd_version=gbd_version)
    supplemental_metadata = load_supplmental_metadata()

    print("Building WPP data...")
    wpp_version = "2024" if out_version == "gbd_2023" else "2022"
    wpp = utils.load_wpp_populations(pm_data, wpp_version=wpp_version)
    # Add GBD location and region ids to the WPP data by mapping on iso3 codes
    wpp = utils.add_gbd_metadata_to_wpp(
        wpp=wpp,
        gbd_hierarchy=hierarchies["gbd"],
        supplemental_metadata=supplemental_metadata,
    )
    # Supplement missing or zero population locations with nans or zeroes respectively
    wpp = utils.add_unmodeled_and_zero_population_locations(
        wpp=wpp,
        supplemental_metadata=supplemental_metadata,
    )
    # Compute the population scalar: the location fraction of the regional population
    wpp["scalar"] = utils.compute_regional_scalar(wpp)

    print("Preparing IHME population data...")
    ihme = utils.prepare_ihme_population(
        populations=populations,
        hierarchies=hierarchies,
        version_tag=version_tag,
    )

    # Use regional scalars from WPP to fill in gaps in IHME data.
    print("Computing missing populations...")
    missing_population = utils.compute_missing_populations(wpp, ihme)

    print("Building raking population...")
    raking_population = utils.build_raking_population(
        ihme_population=ihme,
        wpp_population=wpp,
        missing_population=missing_population,
    )

    print("Building raking shapes...")
    raking_shapes = utils.build_raking_shapes(
        shapes=shapes,
        raking_population=raking_population,
    )

    print("Validating raking data...")
    utils.validate_raking_data(
        raking_population=raking_population,
        raking_shapes=raking_shapes,
    )

    print("Saving raking data...")
    pm_data.save_raking_data(
        population=raking_population,
        shapes=raking_shapes,
        version=f"{out_version}",
    )


@click.command()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_choice(
    "out_version", allow_all=False, choices=["gbd_2023", "gbd_2021", "fhs_2021"]
)
def raking_data(
    output_dir: str,
    out_version: str,
) -> None:
    """Prepare model features."""
    raking_data_main(output_dir, out_version)
