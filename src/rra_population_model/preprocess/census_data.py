import click
import pandas as pd
from rra_tools import jobmon

from rra_population_model.data import PopulationModelData, RRAPopulationData
from rra_population_model import constants as pmc
from rra_population_model import cli_options as clio



def filter_iso3_year_list(
    pop_data: RRAPopulationData,
    iso3_year_list: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Filter out invalid iso3-year combinations from the list."""
    skip_list = [
        ("KEN", "2019"),  # This extraction is not in the right format
        ("HUN", "2022"),  # Still waiting on a GADM file to be processed
    ]
    out_list = []
    for iso3, year in iso3_year_list:
        hard_skip = (iso3, year) in skip_list
        # Need to have at least admin 0 and admin 1 to make plots
        admin_levels = pop_data.list_admin_levels(iso3, year)
        bad_admins = not ({0, 1} < set(admin_levels))
        if not (hard_skip or bad_admins):
            out_list.append((iso3, year))
    return out_list


def census_data_main(
    iso3: str,
    year: str,
    pop_data_dir: str,
    output_directory: str,
) -> None:
    pop_data = RRAPopulationData(pop_data_dir)
    pm_data = PopulationModelData(output_directory)

    # Load census data and shapefiles
    count_data = []
    for measure in ["population_total", "household_total"]:
        print(f"Loading {measure} data for {iso3} {year}")
        try:
            measure_df = pop_data.load_census(measure, iso3, year)
        except FileNotFoundError:
            print(f"Could not find {measure} data for {iso3} {year}")
        measure_data = measure_df.set_index("shape_id")["value"].rename(measure)
        count_data.append(measure_data)
    census_counts = pd.concat(count_data, axis=1).reset_index()

    admin_levels = pop_data.list_admin_levels(iso3, year)
    shapes_by_level = []
    for admin_level in admin_levels:
        print(f"Loading shapefile for {iso3} {year} admin level {admin_level}")
        shapes = pop_data.load_shapefile(admin_level, iso3, year)
        shapes_by_level.append(shapes)
    admin_shapes = pd.concat(shapes_by_level, ignore_index=True)
    admin_shapes = admin_shapes.to_crs(pmc.CRSES["equal_area"].to_string())

    if len(census_counts) != len(admin_shapes):
        msg = f"Length mismatch between census data ({len(census_counts)}) and shapefiles ({len(admin_shapes)})"
        raise ValueError(msg)

    # Merge census with shapefiles
    print(f"Merging census data with shapefiles for {iso3} {year}")
    census_data = admin_shapes.merge(census_counts, how="inner", on="shape_id")

    if len(census_data) != len(admin_shapes):
        msg = f"Length mismatch between merged census data ({len(census_data)}) and shapefiles ({len(admin_shapes)})"
        raise ValueError(msg)

    keep_columns = [
        "shape_id",
        "parent_id",
        "admin_level",
        "shape_name",
        "shape_name_simple",
        "path_to_top_parent",
        "population_total",
        "household_total",
        "geometry",
    ]
    census_data = census_data[keep_columns]

    # Save to output directory
    print(f"Saving census data for {iso3} {year}")
    pm_data.save_admin_training_data(iso3, year, census_data)


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
    pop_data = RRAPopulationData(pop_data_dir)

    census_years = pop_data.list_census_years()
    shapefile_years = pop_data.list_shapefile_years()
    iso3_year_list = list(set(census_years) & set(shapefile_years))

    # Filter out invalid iso3-year combinations
    iso3_year_list = filter_iso3_year_list(pop_data, iso3_year_list)
    if iso3 == clio.RUN_ALL and year == clio.RUN_ALL:
        iso3_and_years = iso3_year_list
    elif iso3 == clio.RUN_ALL:
        iso3_and_years = [(i, y) for i, y in iso3_year_list if y == year]
    elif year == clio.RUN_ALL:
        iso3_and_years = [(i, y) for i, y in iso3_year_list if i == iso3]
    else:
        if (iso3, year) not in iso3_year_list:
            msg = f"Invalid combination of iso3 '{iso3}' and year '{year}'."
            raise ValueError(msg)
        iso3_and_years = [(iso3, year)]

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
    )
