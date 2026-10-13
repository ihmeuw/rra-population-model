import click
import pandas as pd

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    ModelFrameInfo,
    PopulationModelData,
)
from rra_population_model.model_prep.modeling_frame import utils


def modeling_frame_main(
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    bd_data = BuildingDensityData(building_density_dir)
    pm_data = PopulationModelData(output_dir)

    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)

    # Create a modeling frame with the same tile index as the building density data
    modeling_frame = utils.prepare_model_splits(tile_index)
    modeling_frame_info = ModelFrameInfo(
        **tile_index_info.model_dump(),
    )
    pm_data.save_modeling_frame(resolution, modeling_frame, modeling_frame_info)

    # Create a mapping from block_key to iso3/year combinations in our census
    # data that intersect the block.
    block_frame = (
        modeling_frame[["block_key", "geometry"]].dissolve("block_key").reset_index()
    )
    validation_rows = []
    for iso3, year, quarter in pm_data.list_census_data():
        gdf = pm_data.load_census_data(iso3, year, admin_level=0)
        intersection_frame = block_frame[
            block_frame.intersects(gdf.explode().convex_hull.union_all())
        ]
        for block_key in intersection_frame.block_key:
            validation_rows.append((iso3, year, block_key))  # noqa: PERF401
    validation_frame = pd.DataFrame(
        validation_rows, columns=["iso3", "year", "block_key"]
    )
    pm_data.save_validation_frame(resolution, validation_frame)


@click.command()
@clio.with_resolution()
@clio.with_input_directory("building-density", pmc.BUILDING_DENSITY_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def modeling_frame(
    resolution: str,
    building_density_dir: str,
    output_dir: str,
) -> None:
    """Build the global tile index."""
    modeling_frame_main(
        resolution,
        building_density_dir,
        output_dir,
    )
