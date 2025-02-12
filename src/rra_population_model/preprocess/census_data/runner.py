import click
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData, RRAPopulationData
from rra_population_model.preprocess.census_data import utils


def census_data_main(
    iso3: str,
    year: str,
    pop_data_dir: str,
    output_directory: str,
) -> None:
    pop_data = RRAPopulationData(pop_data_dir)
    pm_data = PopulationModelData(output_directory)

    # Load census data and shapefiles
    census_counts = utils.load_census_counts(pop_data, iso3, year)
    census_shapes = utils.load_census_shapes(pop_data, iso3, year)
    census_data = utils.merge_census(census_counts, census_shapes)
    census_data = utils.filter_census_columns(census_data)

    # Save to output directory
    print(f"Saving census data for {iso3} {year}")
    pm_data.save_census_data(census_data, iso3, year)


@click.command()  # type: ignore[arg-type]
@clio.with_iso3()
@clio.with_year()
@clio.with_input_directory("pop-data", pmc.POPULATION_DATA_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def census_data_task(
    iso3: str,
    year: str,
    pop_data_dir: str,
    output_dir: str,
) -> None:
    census_data_main(iso3, year, pop_data_dir, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_iso3(allow_all=True)
@clio.with_year(allow_all=True)
@clio.with_input_directory("pop-data", pmc.POPULATION_DATA_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def census_data(
    iso3: str,
    year: str,
    pop_data_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    iso3_and_years = utils.get_iso3_year_list(pop_data_dir, iso3, year)
    jobmon.run_parallel(
        runner="pmtask preprocess",
        task_name="census_data",
        flat_node_args=(("iso3", "year"), iso3_and_years),
        task_args={
            "pop-data-dir": pop_data_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "75G",
            "runtime": "30m",
            "project": "proj_rapidresponse",
        },
        max_attempts=1,
        log_root=pm_data.log_dir("preprocess-census_data"),
    )
