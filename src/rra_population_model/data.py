from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import shapely
import yaml
from pydantic import BaseModel
from rra_tools.shell_tools import mkdir, touch

from rra_population_model import constants as pmc

if TYPE_CHECKING:
    from rra_population_model.model.modeling.datamodel import ModelSpecification
    from rra_population_model.model.modeling.model import PPSModel

# Type aliases
Polygon: TypeAlias = shapely.Polygon | shapely.MultiPolygon
BBox: TypeAlias = tuple[float, float, float, float]
Bounds: TypeAlias = BBox | Polygon


class TileIndexInfo(BaseModel):
    tile_size: int
    tile_resolution: int
    block_size: int
    crs: str


class BuildingDensityData:
    """Read-only data loader for the building density model."""

    def __init__(self, root: str | Path = pmc.BUILDING_DENSITY_ROOT):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def tiles(self) -> Path:
        return self.root / "tiles"

    def load_tile_index(self, resolution: str) -> gpd.GeoDataFrame:
        self._check_resolution(resolution)
        path = self.tiles / f"{resolution}m" / "tile_index.parquet"
        return gpd.read_parquet(path)

    def load_tile_index_info(self, resolution: str) -> TileIndexInfo:
        self._check_resolution(resolution)
        path = self.tiles / f"{resolution}m" / "tile_index_info.yaml"
        with path.open() as f:
            info = yaml.safe_load(f)
        return TileIndexInfo(**info)

    def tile_path(
        self,
        resolution: str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
    ) -> Path:
        return (
            self.tiles
            / f"{resolution}m"
            / provider
            / time_point
            / f"{block_key}_{measure}.tif"
        )

    def load_tile(
        self,
        resolution: str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
        bounds: shapely.Polygon | None = None,
    ) -> rt.RasterArray:
        self._check_resolution(resolution)
        tile_path = self.tile_path(resolution, provider, block_key, time_point, measure)
        return rt.load_raster(tile_path, bounds=bounds)

    def _check_resolution(self, resolution: str) -> None:
        available_resolutions = [
            p.name for p in self.tiles.iterdir() if p.is_dir() and p.name.endswith("m")
        ]
        if f"{resolution}m" not in available_resolutions:
            msg = f"Resolution {resolution} not available. Available resolutions: {available_resolutions}"
            raise ValueError(msg)


class RRAPopulationData:
    """Read-only data loader for the population model."""

    def __init__(self, root: str | Path = pmc.POPULATION_DATA_ROOT):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def census(self) -> Path:
        return self.root / "census"

    def get_census_dir(
        self,
        iso3: str,
        year: str | int,
    ) -> Path:
        return self.census / iso3 / str(year)

    def list_census_years(self) -> list[tuple[str, str]]:
        """List all available census years by country."""
        return self._list_years(self.census)

    def load_census(
        self,
        census_measure: str,
        iso3: str,
        year: str | int,
    ) -> pd.DataFrame:
        """Load census data from a CSV file.

        Parameters
        ----------
        census_measure
            Census measure to load.
        iso3
            Country ISO3 code.
        year
            Census year.

        Returns
        -------
        pd.DataFrame
            Census data.
        """
        census_dir = self.get_census_dir(iso3, year)
        df = pd.read_csv(census_dir / f"{census_measure}.csv", low_memory=False)
        return df

    @property
    def shapefiles(self) -> Path:
        return self.root / "shapefiles"

    def get_shapefile_dir(
        self,
        iso3: str,
        year: str | int,
        state: str | None = None,
    ) -> Path:
        shapefile_dir = self.shapefiles / iso3 / str(year)
        if state:
            return shapefile_dir / state
        return shapefile_dir

    def list_shapefile_years(self) -> list[tuple[str, str]]:
        """List all available shapefile years by country."""
        return self._list_years(self.shapefiles)

    def load_shapefile(
        self,
        admin_level: int,
        iso3: str,
        year: str | int,
        state: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Load administrative boundary data from a shapefile.

        Parameters
        ----------
        admin_level
            Administrative level to load.
        iso3
            Country ISO3 code.
        year
            The year represented by the shapefile boundaries.
        state
            State or province name. Optional.

        Returns
        -------
        gpd.GeoDataFrame
            Administrative boundary data.
        """
        shape_root = self.get_shapefile_dir(iso3, year, state)
        path = shape_root / f"admin{admin_level}.parquet"
        gdf = gpd.read_parquet(path)
        return gdf

    def list_admin_levels(self, iso3: str, year: str | int) -> list[int]:
        """List all available administrative levels for a country and year."""
        shapefile_dir = self.get_shapefile_dir(iso3, year)
        admin_levels = [
            admin_level
            for admin_level in range(10)
            if (shapefile_dir / f"admin{admin_level}.parquet").exists()
        ]
        return admin_levels

    @property
    def ihme_data(self) -> Path:
        return self.root / "ihme"

    def load_ihme_hierarchy(self) -> pd.DataFrame:
        """Load the global hierarchy data."""
        path = self.ihme_data / "gbd_hierarchy.parquet"
        df = pd.read_parquet(path)
        return df

    def load_ihme_populations(self) -> pd.DataFrame:
        """Load the global population data."""
        path = self.ihme_data / "gbd_population.parquet"
        df = pd.read_parquet(path)
        return df

    def load_ihme_shapes(self) -> gpd.GeoDataFrame:
        """Load the global admin shapefile."""
        path = self.ihme_data / "gbd_shapes.parquet"
        gdf = gpd.read_parquet(path).rename(columns={"loc_id": "location_id"})
        return gdf

    # Helpers

    @staticmethod
    def _list_years(root: Path) -> list[tuple[str, str]]:
        """List all available years by country."""
        years = []
        for iso3_dir in root.iterdir():
            if not iso3_dir.is_dir() or iso3_dir.name == "zzz_logs":
                continue

            for year_dir in iso3_dir.iterdir():
                years.append(  # noqa: PERF401
                    (iso3_dir.name, year_dir.name)
                )
        return years


class ModelFrameInfo(BaseModel):
    tile_size: int
    tile_resolution: int
    block_size: int
    crs: str


class PopulationModelData:
    """Data loader for the population model.

    This class defines a set of relative paths for the population model
    but allows a user to select a different root directory for the model (for
    testing, e.g., or if we need to move to Azure).
    """

    _modeling_frame_filename = "modeling_frame.parquet"
    _modeling_frame_info_filename = "modeling_frame_info.yaml"

    def __init__(self, root: str | Path = pmc.MODEL_ROOT):
        self._root = Path(root)
        self._create_model_root()

    def _create_model_root(self) -> None:
        mkdir(self.root, exist_ok=True)
        mkdir(self.logs, exist_ok=True)

        mkdir(self.admin_inputs, exist_ok=True)
        mkdir(self.census, exist_ok=True)
        mkdir(self.itu_masks, exist_ok=True)
        mkdir(self.raking_inputs, exist_ok=True)
        mkdir(self.gbd_raking_inputs, exist_ok=True)

        mkdir(self.modeling, exist_ok=True)
        for resolution in pmc.RESOLUTIONS.to_list():
            mkdir(self.resolution_root(resolution), exist_ok=True)
            mkdir(self.features_root(resolution), exist_ok=True)
            mkdir(self.training_data_root(resolution), exist_ok=True)
            mkdir(self.tile_training_data_root(resolution), exist_ok=True)
            mkdir(self.model_root(resolution), exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def log_dir(self, step_name: str) -> Path:
        return self.logs / step_name

    @property
    def admin_inputs(self) -> Path:
        return self.root / "admin-inputs"

    @property
    def census(self) -> Path:
        return self.admin_inputs / "census"

    def census_path(self, iso3: str, year: str) -> Path:
        return self.census / f"{iso3}_{year}.parquet"

    def list_census_data(self) -> list[tuple[str, str]]:
        census_data = []
        for path in self.census.glob("*.parquet"):
            # Some iso3 codes have underscores (e.g. GBR subnats)
            *iso3_parts, year = path.stem.split("_")
            iso3 = "_".join(iso3_parts)
            census_data.append((iso3, year))
        return census_data

    def save_census_data(self, gdf: gpd.GeoDataFrame, iso3: str, year: str) -> None:
        path = self.census_path(iso3, year)
        touch(path, clobber=True)
        gdf.to_parquet(path, write_covering_bbox=True)

    def load_census_data(
        self, iso3: str, year: str, bounds: Bounds | None = None
    ) -> gpd.GeoDataFrame:
        bbox = bounds_to_bbox(bounds)
        path = self.census_path(iso3, year)
        return gpd.read_parquet(path, bbox=bbox)

    @property
    def itu_masks(self) -> Path:
        return self.admin_inputs / "itu-masks"

    def itu_mask_path(self, iso3: str) -> Path:
        return self.itu_masks / f"{iso3}.tif"

    def list_itu_iso3s(self) -> list[str]:
        return [f.stem for f in self.itu_masks.glob("*.tif")]

    def load_itu_mask(self, iso3: str) -> rt.RasterArray:
        path = self.itu_mask_path(iso3)
        return rt.load_raster(path)

    @property
    def raking_inputs(self) -> Path:
        return self.admin_inputs / "raking"

    def raking_input_path(self, measure: str, version: str) -> Path:
        return self.raking_inputs / f"{measure}_{version}.parquet"

    def save_raking_data(
        self, population: pd.DataFrame, shapes: gpd.GeoDataFrame, version: str
    ) -> None:
        population_path = self.raking_input_path("population", version)
        touch(population_path, clobber=True)
        population.to_parquet(population_path)

        shapes_path = self.raking_input_path("shapes", version)
        touch(shapes_path, clobber=True)
        shapes.to_parquet(shapes_path, write_covering_bbox=True)

    def load_raking_population(self, version: str) -> pd.DataFrame:
        path = self.raking_input_path("population", version)
        return pd.read_parquet(path)

    def load_raking_shapes(
        self, version: str, bbox: shapely.Polygon | None = None
    ) -> gpd.GeoDataFrame:
        path = self.raking_input_path("shapes", version)
        return gpd.read_parquet(path, bbox=bbox)

    @property
    def gbd_raking_inputs(self) -> Path:
        return self.raking_inputs / "gbd-inputs"

    def gbd_raking_input_path(self, measure: str, version: str) -> Path:
        return self.gbd_raking_inputs / f"{measure}_{version}.parquet"

    def load_gbd_raking_input(
        self,
        measure: str,
        version: str,
        **kwargs: Any,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        path = self.gbd_raking_input_path(measure, version)
        if measure == "shapes":
            return gpd.read_parquet(path, **kwargs)
        return pd.read_parquet(path, **kwargs)

    @property
    def modeling(self) -> Path:
        return self.root / "modeling"

    def resolution_root(self, resolution: str) -> Path:
        return self.modeling / f"{resolution}m"

    def modeling_frame_path(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / self._modeling_frame_filename

    def modeling_frame_info_path(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / self._modeling_frame_info_filename

    def save_modeling_frame(
        self,
        resolution: str,
        model_frame: gpd.GeoDataFrame,
        modeling_frame_info: ModelFrameInfo,
    ) -> None:
        modeling_frame_path = self.modeling_frame_path(resolution)
        mkdir(modeling_frame_path.parent, exist_ok=True)
        touch(modeling_frame_path, clobber=True)
        model_frame.to_parquet(modeling_frame_path)

        modeling_frame_info_path = self.modeling_frame_info_path(resolution)
        touch(modeling_frame_info_path, clobber=True)
        with modeling_frame_info_path.open("w") as f:
            yaml.dump(modeling_frame_info.model_dump(), f)

    def load_modeling_frame(self, resolution: str) -> gpd.GeoDataFrame:
        path = self.modeling_frame_path(resolution)
        return gpd.read_parquet(path)

    def load_modeling_frame_info(self, resolution: str) -> ModelFrameInfo:
        path = self.modeling_frame_info_path(resolution)
        with path.open() as f:
            info = yaml.safe_load(f)
        return ModelFrameInfo(**info)

    def features_root(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / "features"

    def feature_path(
        self,
        resolution: str,
        block_key: str,
        feature_name: str,
        time_point: str,
    ) -> Path:
        return (
            self.features_root(resolution)
            / time_point
            / block_key
            / f"{feature_name}.tif"
        )

    def list_feature_time_points(self, resolution: str) -> list[str]:
        return [p.name for p in self.features_root(resolution).iterdir() if p.is_dir()]

    def link_feature(
        self,
        resolution: str,
        block_key: str,
        feature_name: str,
        time_point: str,
        source_path: Path,
    ) -> None:
        dest = self.feature_path(resolution, block_key, feature_name, time_point)
        mkdir(dest.parent, parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        dest.symlink_to(source_path)

    def save_feature(
        self,
        feature_raster: rt.RasterArray,
        resolution: str,
        block_key: str,
        feature_name: str,
        time_point: str,
    ) -> None:
        out_path = self.feature_path(resolution, block_key, feature_name, time_point)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        touch(out_path, clobber=True)
        save_raster(feature_raster, out_path)

    def feature_exists(
        self,
        resolution: str,
        block_key: str,
        feature_name: str,
        time_point: str,
    ) -> bool:
        path = self.feature_path(resolution, block_key, feature_name, time_point)
        return path.exists()

    def list_features(
        self, resolution: str, block_key: str, time_point: str
    ) -> list[str]:
        block_dir = self.feature_path(resolution, block_key, "", time_point).parent
        return [p.stem for p in block_dir.glob("*.tif")]

    def load_feature(
        self,
        resolution: str,
        block_key: str,
        feature_name: str,
        time_point: str,
        subset_bounds: shapely.Polygon | None = None,
    ) -> rt.RasterArray:
        load_name = feature_name[4:] if feature_name[:3] == "log" else feature_name
        path = self.feature_path(resolution, block_key, load_name, time_point)
        feature = rt.load_raster(path, subset_bounds)
        if feature_name[:3] == "log":
            feature = np.log(1 + feature)
        return feature

    def training_data_root(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / "training-data"

    def tile_training_data_root(self, resolution: str) -> Path:
        return self.training_data_root(resolution) / "tiles"

    def save_tile_training_data(
        self,
        resolution: str,
        tile_key: str,
        tile_gdf: gpd.GeoDataFrame,
        tile_area_weights: gpd.GeoDataFrame,
        tile_rasters: dict[str, rt.RasterArray],
    ) -> None:
        root = self.tile_training_data_root(resolution) / tile_key
        mkdir(root, exist_ok=True)

        gdf_path = root / "people_per_structure.parquet"
        touch(gdf_path, clobber=True)
        tile_gdf.to_parquet(gdf_path)

        paw_path = root / "pixel_area_weights.parquet"
        touch(paw_path, clobber=True)
        tile_area_weights.to_parquet(paw_path)

        for measure, raster in tile_rasters.items():
            raster_path = root / f"{measure}.tif"
            touch(raster_path, clobber=True)
            save_raster(raster, raster_path)

    def save_summary_people_per_structure(
        self,
        data: pd.DataFrame,
        resolution: str,
    ) -> None:
        root = self.training_data_root(resolution)
        path = root / "people_per_structure.parquet"
        touch(path, clobber=True)
        data.to_parquet(path)

    def load_people_per_structure(
        self, resolution: str, tile_key: str | None = None
    ) -> gpd.GeoDataFrame:
        if tile_key:
            root = self.tile_training_data_root(resolution) / tile_key
        else:
            root = self.training_data_root(resolution)
        return gpd.read_parquet(root / "people_per_structure.parquet")

    def load_pixel_area_weights(
        self, resolution: str, tile_key: str | None = None
    ) -> gpd.GeoDataFrame:
        if tile_key:
            root = self.tile_training_data_root(resolution) / tile_key
        else:
            root = self.training_data_root(resolution)
        return pd.read_parquet(root / "pixel_area_weights.parquet")

    def list_tile_training_data(self, resolution: str) -> list[str]:
        return [
            p.stem
            for p in self.tile_training_data_root(resolution).glob("*")
            if p.is_dir()
        ]

    def load_tile_training_data(
        self,
        resolution: str,
        tile_key: str,
        measure: str,
    ) -> rt.RasterArray:
        path = self.tile_training_data_root(resolution) / tile_key / f"{measure}.tif"
        return rt.load_raster(path)

    def model_root(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / "models"

    def model_version_root(self, resolution: str, version: str) -> Path:
        return self.model_root(resolution) / version

    def make_model_version_root(self, resolution: str, version: str) -> Path:
        dirs = [
            self.model_version_root(resolution, version),
            self.raw_predictions_root(resolution, version),
            self.raking_factors_root(resolution, version),
            self.raked_predictions_root(resolution, version),
            self.compiled_predictions_root(resolution, version),
        ]
        for path in dirs:
            mkdir(path)

        return path

    def save_model_specification(
        self,
        model_spec: "ModelSpecification",
    ) -> None:
        self.make_model_version_root(model_spec.resolution, model_spec.model_version)
        path = Path(model_spec.output_root) / "specification.yaml"
        touch(path)
        with path.open("w") as f:
            yaml.safe_dump(model_spec.model_dump(mode="json"), f)

    def load_model_specification(
        self, resolution: str, version: str
    ) -> "ModelSpecification":
        from rra_population_model.model.modeling.datamodel import ModelSpecification

        path = self.model_version_root(resolution, version) / "specification.yaml"
        with path.open() as f:
            spec = yaml.safe_load(f)
        return ModelSpecification.model_validate(spec)

    def load_model(self, resolution: str, version: str) -> "PPSModel":
        from rra_population_model.model.modeling.model import PPSModel

        ckpt_path = self.model_version_root(resolution, version) / "best_model.ckpt"
        return PPSModel.load_from_checkpoint(ckpt_path)

    def raw_predictions_root(self, resolution: str, version: str) -> Path:
        return self.model_version_root(resolution, version) / "raw_predictions"

    def raw_prediction_path(
        self,
        block_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> Path:
        resolution = model_spec.resolution
        version = model_spec.model_version
        return (
            self.raw_predictions_root(resolution, version)
            / time_point
            / f"{block_key}.tif"
        )

    def list_raw_prediction_time_points(
        self, resolution: str, version: str
    ) -> list[str]:
        return [
            p.name
            for p in self.raw_predictions_root(resolution, version).iterdir()
            if p.is_dir()
        ]

    def save_raw_prediction(
        self,
        raster: rt.RasterArray,
        block_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> None:
        path = self.raw_prediction_path(block_key, time_point, model_spec)
        mkdir(path.parent, exist_ok=True)
        save_raster(raster, path)

    def load_raw_prediction(
        self,
        block_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> rt.RasterArray:
        path = self.raw_prediction_path(block_key, time_point, model_spec)
        return rt.load_raster(path)

    def raking_factors_root(self, resolution: str, version: str) -> Path:
        return self.model_version_root(resolution, version) / "raking_factors"

    def raking_factor_path(
        self, time_point: str, model_spec: "ModelSpecification"
    ) -> Path:
        return (
            self.raking_factors_root(model_spec.resolution, model_spec.model_version)
            / f"{time_point}.parquet"
        )

    def list_raking_factor_time_points(
        self, resolution: str, version: str
    ) -> list[str]:
        return [
            p.stem
            for p in self.raking_factors_root(resolution, version).iterdir()
            if p.is_file()
        ]

    def save_raking_factors(
        self,
        raking_factors: gpd.GeoDataFrame,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> None:
        path = self.raking_factor_path(time_point, model_spec)
        touch(path, clobber=True)
        raking_factors.to_parquet(path)

    def load_raking_factors(
        self,
        time_point: str,
        model_spec: "ModelSpecification",
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        path = self.raking_factor_path(time_point, model_spec)
        return gpd.read_parquet(path, **kwargs)

    def raked_predictions_root(self, resolution: str, version: str) -> Path:
        return self.model_version_root(resolution, version) / "raked_predictions"

    def raked_prediction_path(
        self, block_key: str, time_point: str, model_spec: "ModelSpecification"
    ) -> Path:
        resolution = model_spec.resolution
        version = model_spec.model_version
        return (
            self.raked_predictions_root(resolution, version)
            / time_point
            / f"{block_key}.tif"
        )

    def list_raked_prediction_time_points(
        self, resolution: str, version: str
    ) -> list[str]:
        return [
            p.name
            for p in self.raked_predictions_root(resolution, version).iterdir()
            if p.is_dir()
        ]

    def save_raked_prediction(
        self,
        raster: rt.RasterArray,
        block_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> None:
        path = self.raked_prediction_path(block_key, time_point, model_spec)
        mkdir(path.parent, exist_ok=True)
        save_raster(raster, path)

    def load_raked_prediction(
        self,
        block_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> rt.RasterArray:
        path = self.raked_prediction_path(block_key, time_point, model_spec)
        return rt.load_raster(path)

    def compiled_predictions_root(self, resolution: str, version: str) -> Path:
        return self.model_version_root(resolution, version) / "compiled_predictions"

    def compiled_prediction_path(
        self, group_key: str, time_point: str, model_spec: "ModelSpecification"
    ) -> Path:
        resolution = model_spec.resolution
        version = model_spec.model_version
        return (
            self.compiled_predictions_root(resolution, version)
            / time_point
            / f"{group_key}.tif"
        )

    def compiled_prediction_vrt_path(
        self, time_point: str, model_spec: "ModelSpecification"
    ) -> Path:
        resolution = model_spec.resolution
        version = model_spec.model_version
        return (
            self.compiled_predictions_root(resolution, version)
            / time_point
            / "index.vrt"
        )

    def list_compiled_prediction_time_points(
        self, resolution: str, version: str
    ) -> list[str]:
        return [
            p.name
            for p in self.compiled_predictions_root(resolution, version).iterdir()
            if p.is_dir()
        ]

    def save_compiled_prediction(
        self,
        raster: rt.RasterArray,
        group_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
        **save_kwargs: Any,
    ) -> None:
        path = self.compiled_prediction_path(group_key, time_point, model_spec)
        mkdir(path.parent, exist_ok=True)
        save_raster_to_cog(raster, path, **save_kwargs)

    def load_compiled_prediction(
        self,
        group_key: str,
        time_point: str,
        model_spec: "ModelSpecification",
    ) -> rt.RasterArray:
        path = self.compiled_prediction_path(group_key, time_point, model_spec)
        return rt.load_raster(path)

    @property
    def results(self) -> Path:
        return Path(self.root, "results")

    @property
    def figure_results(self) -> Path:
        return Path(self.root / "figure_results")

    @property
    def itu(self) -> Path:
        return Path(self.root, "itu")

    @property
    def itu_results(self) -> Path:
        return self.itu / "results"

    def save_itu_results(
        self, raster: rt.RasterArray, iso3: str, model_spec: "ModelSpecification"
    ) -> None:
        resolution = model_spec.resolution
        model_name = model_spec.model_version
        out_path = self.itu_results / f"{resolution}m" / model_name / f"{iso3}.tif"
        mkdir(out_path.parent, exist_ok=True, parents=True)
        save_raster(raster, out_path)


def bounds_to_bbox(bounds: Bounds | None) -> BBox | None:
    if isinstance(bounds, Polygon):
        bbox = bounds.bounds
    elif isinstance(bounds, tuple):
        bbox = bounds
    elif bounds is None:
        bbox = None
    else:
        msg = f"Invalid bounds type: {type(bounds)}"
        raise TypeError(msg)
    return bbox  # type: ignore[no-any-return]


def save_raster(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    **kwargs: Any,
) -> None:
    """Save a raster to a file with standard parameters."""
    save_params = {
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "ZSTD",
        "predictor": 2,  # horizontal differencing
        "num_threads": num_cores,
        "bigtiff": "yes",
        **kwargs,
    }
    touch(output_path, clobber=True)
    raster.to_file(output_path, **save_params)


def save_raster_to_cog(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    resampling: str = "nearest",
) -> None:
    """Save a raster to a COG file."""
    cog_save_params = {
        "driver": "COG",
        "overview_resampling": resampling,
    }
    save_raster(raster, output_path, num_cores, **cog_save_params)
