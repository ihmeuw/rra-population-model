import click
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    ModelFrameInfo,
    PopulationModelData,
)


def prepare_model_splits(modeling_frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Generate a set of seeds we can use to make consistent test/train splits
    rs = np.random.RandomState(42)
    seeds = rs.randint(0, 2**32 - 1, size=(10, 2))

    def _select(tiles: list[str]) -> "pd.Series[bool]":
        return modeling_frame.tile_key.isin(tiles)  # type: ignore[no-any-return]

    test_size = 0.2
    validate_size = 0.1

    for i, (test_seed, validate_seed) in enumerate(seeds):  # type: ignore[misc]
        # 20% of the overall data is used for testing.

        train_tiles, test_tiles = train_test_split(
            modeling_frame.tile_key.tolist(),
            test_size=test_size,
            random_state=test_seed,  # type: ignore[has-type]
        )
        # 10% of the overall data is used for validation/hyper-parameter tuning.
        train_tiles, validate_tiles = train_test_split(
            train_tiles,
            test_size=validate_size * (1 - test_size),
            random_state=validate_seed,  # type: ignore[has-type]
        )

        modeling_frame[f"split_{i}"] = "train"
        modeling_frame.loc[_select(test_tiles), f"split_{i}"] = "test"
        modeling_frame.loc[_select(validate_tiles), f"split_{i}"] = "validate"

    return modeling_frame


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
    modeling_frame = prepare_model_splits(tile_index)
    modeling_frame_info = ModelFrameInfo(
        **tile_index_info.model_dump(),
    )
    pm_data.save_modeling_frame(resolution, modeling_frame, modeling_frame_info)


@click.command()  # type: ignore[arg-type]
@clio.with_resolution(pmc.RESOLUTIONS)
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
