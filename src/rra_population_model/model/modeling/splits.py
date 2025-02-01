"""Functions for generating train/validate/test splits.

When the modeling frame is generated, we reproducibly split the tiles for the
entire world into training, validation, and test sets. For actual modeling, our
training data is limited to country-years where we have good ground truth census
data. This module provides functions for loading the modeling frame, filtering it
to the training data, and generating the train/validate/test splits for the model.
"""

from pathlib import Path

import geopandas as gpd

from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.model.modeling.datamodel import (
    ModelPartition,
    ModelSplit,
    TileID,
)


def get_train_validate_test_splits(
    model_root: str | Path,
    resolution: str,
) -> dict[int, ModelPartition]:
    """Get the train/validate/test splits for the model.

    Parameters
    ----------
    model_root
        The root directory of the people per structure model.
    data_root
        The root directory of the RRA population data.
    **parallel_kwargs
        Keyword arguments to pass to the parallel runner.

    Returns
    -------
    dict[int, ModelSplit]
        A dictionary of the splits, keyed by the split number.
    """
    training_frame = _load_training_frame(
        model_root=model_root,
        resolution=resolution,
    )
    splits = {}
    for split in range(10):
        partition = []
        for split_var in ["train", "validate", "test"]:
            mask = training_frame[f"split_{split}"] == split_var
            model_split = ModelSplit()
            for row in training_frame.loc[mask].itertuples():
                model_split.append(
                    TileID(row.block_key, row.tile_key, row.geometry, "2020q1")
                )
            partition.append(model_split)
        splits[split] = ModelPartition(*partition)
    return splits


###########
# Helpers #
###########


def _load_training_frame(
    model_root: str | Path,
    resolution: str,
) -> gpd.GeoDataFrame:
    """Load the training frame for the model.

    The training frame is the subset of the modeling frame that contains valid
    training data.  For now, it's just MEX 2020, but we'll add more country-years
    as we go.

    Parameters
    ----------
    model_root
        The root directory of the people per structure model.
    resolution
        The resolution of the model data.

    Returns
    -------
    gpd.GeodataFrame
        The training frame with columns containing the names of the tiles,
        polygon boundaries for those tiles, and several `split` indicator
        columns labeling each tile as `train`, `validate`, or `test`, which
        we can use for cross-validation.
    """
    pm_data = PopulationModelData(model_root)
    modeling_frame = pm_data.load_modeling_frame(resolution)
    training_tiles = pm_data.list_tile_training_data()
    training_frame = modeling_frame[modeling_frame.tile_key.isin(training_tiles)]
    return training_frame
