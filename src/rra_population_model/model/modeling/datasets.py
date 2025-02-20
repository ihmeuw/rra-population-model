"""This module contains torch-compatible datasets for the people per structure model."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import lightning
import numpy as np
import pandas as pd
import rasterra as rt
import torch
from rra_tools import parallel
from torch.utils.data import DataLoader, Dataset, default_collate

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model.modeling.datamodel import (
    ModelSpecification,
    ModelSplit,
    TileID,
)
from rra_population_model.model.modeling.splits import (
    get_train_validate_test_splits,
)

##############
# Shape data #
##############
#
# Shape data is training data aggregated at the administrative unit level. It includes
# several potential training targets (e.g., occupancy rate, population density) and
# features (e.g., night-time lights, building density averages).
# This is one potential way we might structure the data for training the model.


def load_shape_data(
    resolution: str,
    tile_keys: list[TileID],
    model_root: str | Path,
    denominator: str,
    features: list[str],
    **parallel_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Load the shape data for the given tiles.

    Parameters
    ----------
    tile_keys
        The tiles to load data for.
    model_root
        The root directory of the people per structure model.
    denominator
        The denominator to use for the occupancy rate.
    features
        The features to include in the model.
    **parallel_kwargs
        Keyword arguments to pass to the parallel runner.

    Returns
    -------
    gpd.GeoDataFrame
        The shape data for the given tiles.
    """
    pm_data = PopulationModelData(model_root)
    admin_data = pm_data.load_people_per_structure(resolution)

    in_tiles = admin_data.tile_key.isin([tk.tile_key for tk in tile_keys])
    # For now we're only using data from admins that have both buildings
    # and people in them.  We'll consider the 0 occupancy rate case later.
    positive_occupancy = admin_data[f"admin_occupancy_rate_{denominator}"] > 0.0
    single_tile = ~admin_data["admin_multi_tile"].astype(bool)

    mask = in_tiles & positive_occupancy & single_tile
    admin_data = (
        admin_data.loc[mask]
        .rename(columns=lambda x: x[6:] if x[:6] == "admin_" else x)
        .rename(
            columns={
                "id": "admin_id",
                f"occupancy_rate_{denominator}": "occupancy_rate",
                f"log_occupancy_rate_{denominator}": "log_occupancy_rate",
                f"built_{denominator}": "built",
                f"population_{denominator}": "population",
            }
        )
    )

    keep_cols = [
        "admin_id",
        "block_key",
        "tile_key",
        "time_point",
        "occupancy_rate",
        "log_occupancy_rate",
        "built",
        "population",
        "geometry",
        *features,
    ]
    return admin_data[keep_cols]


##############
# Pixel data #
##############
#
# Pixel data is training data aggregated/disaggregated (depending on the data) to the
# pixel level. It has the same potential training targets and features as the shape
# data, just at a regular gridded resolution. This is another potential way we might
# structure the data for training the model as it mirrors how we'll do inference.


def load_pixel_data(
    tile_keys: list[TileID],
    model_root: str | Path,
    resolution: str,
    denominator: str,
    features: list[str],
    **parallel_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pixel data for the given list of tiles.

    Parameters
    ----------
    tile_keys
        The tiles to load data for.
    model_root
        The root directory of the people per structure model.
    **parallel_kwargs
        Keyword arguments to pass to the parallel runner.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The pixel data and pixel area weights for the given tiles.
        The area weights tell us what proportion of each pixel's area is
        in each administrative unit.
    """
    tile_specs = [
        (tk, model_root, resolution, denominator, features) for tk in tile_keys
    ]

    _load_tile_pixels(tile_specs[0])
    pixel_data = parallel.run_parallel(
        _load_tile_pixels,
        tile_specs,
        **parallel_kwargs,
    )
    data_, area_weights_ = zip(*pixel_data, strict=False)
    data = pd.concat([d for d in data_ if not d.empty], ignore_index=True)
    area_weights = pd.concat(
        [aw for aw in area_weights_ if not aw.empty], ignore_index=True
    )

    data["pixel_id"] = data["pixel_id"].astype(int)
    data = data.set_index(["tile_key", "pixel_id"]).sort_index().reset_index()

    area_weights["pixel_id"] = area_weights["pixel_id"].astype(int)
    area_weights = (
        area_weights.set_index(["tile_key", "pixel_id"]).sort_index().reset_index()
    )

    return data, area_weights


def _load_tile_pixels(
    tile_specification: tuple[TileID, str | Path, str, str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pixel data for a given tile.

    Parameters
    ----------
    tile_specification
        A tuple containing the tile key, time point, model root, denominator, and features.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The pixel data and pixel area weights for the given tile.
        The area weights tell us what proportion of each pixel's area is
        in each administrative unit.
    """
    tile, model_root, resolution, denominator, features = tile_specification
    pm_data = PopulationModelData(model_root)

    tile_features = {}
    for feature_name in features:
        tile_features[feature_name] = pm_data.load_feature(
            resolution,
            tile.block_key,
            feature_name,
            tile.time_point,
            tile.tile_poly,
        )

    built = pm_data.load_feature(
        resolution,
        tile.block_key,
        denominator,
        tile.time_point,
        tile.tile_poly,
    )
    population = pm_data.load_tile_training_data(
        resolution,
        tile.tile_key,
        f"population_{denominator}",
    )

    occupancy_rate = pm_data.load_tile_training_data(
        resolution,
        tile.tile_key,
        f"occupancy_rate_{denominator}",
    )
    log_occupancy_rate = pm_data.load_tile_training_data(
        resolution,
        tile.tile_key,
        f"log_occupancy_rate_{denominator}",
    )
    multi_tile = pm_data.load_tile_training_data(
        resolution,
        tile.tile_key,
        "multi_tile",
    )

    tile_data = {
        "built": built,
        "population": population,
        "occupancy_rate": occupancy_rate,
        "log_occupancy_rate": log_occupancy_rate,
        "multi_tile": multi_tile,
        **tile_features,
    }
    area_weights = pm_data.load_pixel_area_weights(resolution, tile.tile_key).set_index(
        "pixel_id"
    )

    def extract(key: str) -> pd.Series:  # type: ignore[type-arg]
        return pd.Series(tile_data[key].to_numpy().flatten(), name=key)

    occ_rate_threshold = 1e-3
    is_good = extract("occupancy_rate") > occ_rate_threshold
    single_tile = ~extract("multi_tile").astype(bool)

    mask = is_good & single_tile

    if mask.sum() > 0:
        area_weights = area_weights[mask.loc[area_weights.index]].reset_index()
        area_weights["tile_key"] = tile.tile_key

        pixel_data = pd.concat([extract(k)[mask] for k in tile_data], axis=1)
        pixel_data.index.name = "pixel_id"
        pixel_data = pixel_data.reset_index()
        pixel_data["pixel_id"] = pixel_data["pixel_id"].astype(int)
        pixel_data["tile_key"] = tile.tile_key
    else:
        area_weights = pd.DataFrame(columns=area_weights.columns)
        pixel_data = pd.DataFrame(columns=list(tile_data))

    return pixel_data, area_weights


class PPSDataset(Dataset[dict[str, torch.Tensor]]):
    """A torch-compatible dataset for the people per structure model."""

    def __init__(
        self,
        resolution: str,
        tiles: ModelSplit,
        features: list[str],
        model_root: str | Path = pmc.MODEL_ROOT,
        data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
        **parallel_kwargs: Any,
    ) -> None:
        self._tiles = tiles
        self._features = features

        if data is None:
            shape_data = load_shape_data(
                resolution,
                tiles,
                model_root=model_root,
                **parallel_kwargs,
            )
            pixel_data, area_weights = load_pixel_data(
                tiles,
                model_root=model_root,
                **parallel_kwargs,
            )
        else:
            shape_data, pixel_data, area_weights = data

        self._shape_data = shape_data
        self._pixel_data = pixel_data
        self._area_weights = area_weights

        self._lhs_tensors = {}
        for col_name in [
            "built",
            "population",
            "occupancy_rate",
            "log_occupancy_rate",
        ]:
            self._lhs_tensors[f"admin_{col_name}"] = torch.tensor(
                self._shape_data.loc[:, [col_name]].values,
                dtype=torch.float32,
            )
            self._lhs_tensors[f"pixel_{col_name}"] = torch.tensor(
                self._pixel_data.loc[:, [col_name]].values,
                dtype=torch.float32,
            )
        self._shape_feature_tensor = torch.tensor(
            self._shape_data.loc[:, self._features].values,
            dtype=torch.float32,
        )
        self._pixel_feature_tensor = torch.tensor(
            self._pixel_data.loc[:, self._features].values,
            dtype=torch.float32,
        )

        # Create a sparse tensor for the area weights.
        # This tensor will have shape (num_admins, num_pixels) and
        # will tell us what proportion of each pixel's area is in each
        # administrative unit.
        tiles_and_pixels = self._pixel_data.set_index(["tile_key", "pixel_id"]).index
        col_map = pd.Series(range(len(tiles_and_pixels)), index=tiles_and_pixels)
        cols = col_map.loc[self._area_weights.set_index(["tile_key", "pixel_id"]).index]
        admins = self._area_weights["admin_id"].unique()
        row_map = pd.Series(range(len(admins)), index=admins)
        rows = row_map.loc[self._area_weights.set_index("admin_id").index]

        self._area_weight_tensor = torch.sparse_coo_tensor(
            np.array([rows.to_numpy(), cols.to_numpy()]),  # type: ignore[arg-type]
            self._area_weights["admin_area_weight"].to_numpy(),  # type: ignore[arg-type]
        )

    def __len__(self) -> int:
        return 1

    @property
    def num_features(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            **self._lhs_tensors,
            "admin_features": self._shape_feature_tensor,
            "pixel_features": self._pixel_feature_tensor,
            "pixel_area_weights": self._area_weight_tensor,
        }


class InferenceDataset(Dataset[dict[str, Any]]):
    """Inference dataset for the people per structure model.

    Inference is done on tiles from the model frame, which are a stack of 2D
    feature rasters. This dataset is used to load the features for a given
    tile and time point.

    """

    def __init__(
        self,
        resolution: str,
        block_keys: list[str],
        time_point: str,
        denominator: str,
        features: list[str],
        model_root: str | Path = pmc.MODEL_ROOT,
    ) -> None:
        self._resolution = resolution
        self._block_keys = block_keys
        self._time_point = time_point
        self._denominator = denominator
        self._features = features
        self._pm_data = PopulationModelData(model_root)

    def __len__(self) -> int:
        return len(self._block_keys)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Load the features for the given tile and time point.
        # These come back as a dict of feature names to RasterArrays.
        block_key = self._block_keys[idx]
        tile_data = {}
        for feature_name in self._features:
            tile_data[feature_name] = self._pm_data.load_feature(
                self._resolution,
                block_key,
                feature_name,
                self._time_point,
            )

        tile_data["built"] = self._pm_data.load_feature(
            self._resolution,
            block_key,
            self._denominator,
            self._time_point,
        )

        # Subset the features we want and convert the whole thing to
        # a tensor with the correct shape
        feature_tensor = torch.stack(
            [
                torch.tensor(tile_data[feature].to_numpy(), dtype=torch.float32)
                for feature in self._features
            ],
            dim=-1,
        )
        built = tile_data["built"]
        built_tensor = torch.tensor(built.to_numpy(), dtype=torch.float32)

        return {
            "block": f"{block_key}_{self._time_point}",
            "features": feature_tensor,
            "built": built_tensor,
            "raster_template": built,
        }


#########################
# Lightning data module #
#########################


class PPSDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        model_specification: dict[str, Any],
        *,
        verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.specification = ModelSpecification(**model_specification)

        partitions = get_train_validate_test_splits(
            model_root=self.specification.model_root,
            resolution=self.specification.resolution,
        )
        self.model_partition = partitions[self.specification.split]
        self.denominator = self.specification.denominator
        self.features = self.specification.features
        self.model_root = self.specification.model_root

        self.datasets: dict[str, PPSDataset] = {}

        self.verbose = verbose

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up the data module.

        This method is called once per process before the training loop.
        """
        if not self.datasets:
            tile_keys = self.model_partition.all_tile_keys
            shape_data = load_shape_data(
                self.specification.resolution,
                tile_keys,
                self.model_root,
                self.denominator,
                self.features,
                num_cores=self.specification.num_cores,
                progress_bar=self.verbose,
            )
            pixel_data, area_weights = load_pixel_data(
                tile_keys,
                self.model_root,
                self.specification.resolution,
                self.denominator,
                self.features,
                num_cores=self.specification.num_cores,
                progress_bar=self.verbose,
            )

            for split_name, split in self.model_partition._asdict().items():
                data = (
                    filter_tiles(shape_data, split),
                    filter_tiles(pixel_data, split),
                    filter_tiles(area_weights, split),
                )
                dataset = PPSDataset(
                    self.specification.resolution,
                    tiles=split,
                    features=self.features,
                    model_root=self.model_root,
                    data=data,
                )
                self.datasets[split_name] = dataset

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Get the training dataloader."""
        return DataLoader(self.datasets["train"], collate_fn=custom_collate)

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Get the validation dataloader."""
        return DataLoader(self.datasets["validate"], collate_fn=custom_collate)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Get the test dataloader."""
        return DataLoader(self.datasets["test"], collate_fn=custom_collate)


class InferenceDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        model_specification: dict[str, Any],
        block_keys: list[str],
        time_point: str,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.specification = ModelSpecification(**model_specification)
        self.resolution = self.specification.resolution
        self.block_keys = block_keys
        self.time_point = time_point
        self.features = self.specification.features
        self.model_root = self.specification.model_root

        self.num_workers = num_workers

        self.dataset: InferenceDataset

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up the data module.

        This method is called once per process before the training loop.
        """
        self.dataset = InferenceDataset(
            resolution=self.resolution,
            block_keys=self.block_keys,
            time_point=self.time_point,
            denominator=self.specification.denominator,
            features=self.features,
            model_root=self.model_root,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Get the prediction dataloader."""
        return DataLoader(
            self.dataset, collate_fn=custom_collate, num_workers=self.num_workers
        )


def filter_tiles(df: pd.DataFrame, split: ModelSplit) -> pd.DataFrame:
    """Filter a dataframe to only include data for a list of tiles."""
    return df[df["tile_key"].isin(split.tile_keys)]


def is_sparse_tensor(element: Any) -> bool:
    """Check if an element is a sparse tensor."""
    sparse_layouts = {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }
    return isinstance(element, torch.Tensor) and element.layout in sparse_layouts


def custom_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Custom collate function for the people-per-structure model.

    This function lets us define how non-standard data types are collated
    into batches. In this case, we want to avoid collating sparse tensors
    as they are not supported by the default collate function.

    Parameters
    ----------
    batch
        A list of dictionaries, where each dictionary contains a batch of data.

    Returns
    -------
    dict[str, torch.Tensor | list[torch.Tensor]]
        A dictionary containing the collated batch.
    """
    zip_dict = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            zip_dict[k].append(v)

    out_dict: dict[str, Any] = {}
    for k, elements in zip_dict.items():
        if is_sparse_tensor(elements[0]) or isinstance(elements[0], rt.RasterArray):
            out_dict[k] = elements
        else:
            out_dict[k] = default_collate(elements)
    return out_dict
