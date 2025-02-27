import click

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
