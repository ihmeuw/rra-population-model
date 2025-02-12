import geopandas as gpd
import pandas as pd

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import RRAPopulationData

##################
# Arg processing #
##################


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


def get_iso3_year_list(
    pop_data_dir: str, iso3: str, year: str
) -> list[tuple[str, str]]:
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

    return iso3_and_years


###########
# Loaders #
###########


def load_census_counts(
    pop_data: RRAPopulationData, iso3: str, year: str
) -> pd.DataFrame:
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
    return census_counts


def load_census_shapes(
    pop_data: RRAPopulationData, iso3: str, year: str
) -> gpd.GeoDataFrame:
    admin_levels = pop_data.list_admin_levels(iso3, year)
    shapes_by_level = []
    for admin_level in admin_levels:
        print(f"Loading shapefile for {iso3} {year} admin level {admin_level}")
        shapes = pop_data.load_shapefile(admin_level, iso3, year)
        shapes_by_level.append(shapes)
    admin_shapes = pd.concat(shapes_by_level, ignore_index=True)
    admin_shapes = admin_shapes.to_crs(pmc.CRSES["equal_area"].to_pyproj())
    return admin_shapes


def merge_census(
    census_counts: pd.DataFrame, census_shapes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if len(census_counts) != len(census_shapes):
        msg = f"Length mismatch between census data ({len(census_counts)}) and shapefiles ({len(census_shapes)})"
        raise ValueError(msg)

    # Merge census with shapefiles
    census_data = census_shapes.merge(census_counts, how="inner", on="shape_id")

    if len(census_data) != len(census_shapes):
        msg = f"Length mismatch between merged census data ({len(census_data)}) and shapefiles ({len(census_shapes)})"
        raise ValueError(msg)

    return census_data


def filter_census_columns(census_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    census_data = census_data.loc[:, keep_columns]
    return census_data
