import abc
from pathlib import Path
from typing import Any

import numpy as np
import rasterra as rt

from rra_population_model import constants as pmc
from rra_population_model.data import (
    BuildingDensityData,
    PopulationModelData,
)
from rra_population_model.model_prep.features import obm, utils
from rra_population_model.model_prep.features.metadata import FeatureMetadata
from rra_population_model.model_prep.features.msft_obm import splice_p


class ProcessingStrategy:
    def __init__(
        self,
        built_version: pmc.BuiltVersion,
        feature_metadata: FeatureMetadata,
    ) -> None:
        self.built_version = built_version
        self.feature_metadata = feature_metadata

    @abc.abstractmethod
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def generate_derived_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def generate_geospatial_averages(
        self,
        features: list[str],
        feature_average_radii: list[int],
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        pass

    @abc.abstractmethod
    def link_features(
        self,
        feature_paths: dict[str, Path],
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,
        pm_data: PopulationModelData,
    ) -> None:
        pass


class SkipStrategy(ProcessingStrategy):
    def generate_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def generate_derived_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def generate_geospatial_averages(
        self,
        features: list[str],  # noqa: ARG002
        feature_average_radii: list[int],  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> dict[str, Path]:
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )
        return {}

    def link_features(
        self,
        feature_paths: dict[str, Path],  # noqa: ARG002
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> None:
        assert fill_time_points == []  # noqa: S101
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )


class ProcessStrategy(ProcessingStrategy):
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        out_paths = {}
        for measure in self.built_version.measures:
            out_measure = f"{self.built_version.name}_{measure}"
            print(f"Processing {measure} for {self.built_version.name}.")
            source_path = bd_data.tile_path(
                provider=self.built_version.name,
                measure=measure,
                **self.feature_metadata.shared_kwargs,
            )
            if not source_path.exists():
                msg = f"Source path {source_path} does not exist."
                raise FileNotFoundError(msg)
            print(f"Linking {measure} for {self.built_version.name}.")
            pm_data.link_feature(
                source_path=source_path,
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
            out_paths[out_measure] = source_path
        return out_paths

    def generate_derived_measures(
        self,
        bd_data: BuildingDensityData,  # noqa: ARG002
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        # Some versions ship every measure they have and derive nothing. Listing
        # them explicitly keeps the fall-through loud: an unrecognised version is
        # a mistake, not a version with no derived measures.
        if self.built_version.name in NO_DERIVED_MEASURES:
            return {}
        if self.built_version.name not in FEATURE_SPECS:
            msg = f"Unknown built version: {self.built_version.name}"
            raise ValueError(msg)
        return _generate_provider_derived_measures(
            pm_data, self.feature_metadata, self.built_version.name
        )

    def generate_geospatial_averages(
        self,
        features: list[str],
        feature_average_radii: list[int],
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        out_paths = {}
        for feature in features:
            for radius in feature_average_radii:
                print(f"Processing {feature} with radius {radius}m.")
                buffered_measure = mosaic_tile(
                    measure=feature,
                    feature_metadata=self.feature_metadata,
                    pm_data=pm_data,
                )
                print(f"Processing {feature} with radius {radius}m.")
                average_measure = (
                    utils.make_spatial_average(
                        tile=buffered_measure,
                        radius=radius,
                        kernel_type="gaussian",
                    )
                    .resample_to(self.feature_metadata.block_template, "average")
                    .astype(np.float32)
                )
                pm_data.save_feature(
                    average_measure,
                    feature_name=f"{feature}_{radius}m",
                    **self.feature_metadata.shared_kwargs,
                )

                out_paths[f"{feature}_{radius}m"] = pm_data.feature_path(
                    feature_name=f"{feature}_{radius}m",
                    **self.feature_metadata.shared_kwargs,
                )
        return out_paths

    def link_features(
        self,
        feature_paths: dict[str, Path],
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,
        pm_data: PopulationModelData,
    ) -> None:
        for feature, path in feature_paths.items():
            for time_point in fill_time_points:
                pm_data.link_feature(
                    source_path=path,
                    feature_name=feature,
                    time_point=time_point,
                    block_key=feature_metadata.block_key,
                    resolution=feature_metadata.resolution,
                )


class ObmStrategy(ProcessStrategy):
    """Build OBM's own measures from the occupancy covariate.

    `ProcessStrategy.generate_measures` links ready-made density and height
    tiles out of the building-density layout. OBM has neither: its covariate is
    eight occupancy-class rasters under a different root, and its density is
    their sum after rescaling the pixels where OSM building relations
    double-count the same ground. Height is GHSL's ANBH with one storey imputed
    where OBM sees a footprint and GHSL sees nothing.

    Everything after that is inherited unchanged - in particular the symlink
    fan-out, which for a single-epoch version fills all 69 other time points.
    """

    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        block_key = self.feature_metadata.block_key
        resolution = self.feature_metadata.resolution
        covariate_root = (
            pm_data.open_building_map_covariates / f"{resolution}m" / block_key
        )
        if not covariate_root.exists():
            print(f"No OBM coverage for block {block_key}; skipping.")
            return {}

        print(f"Loading OBM parent densities for {block_key}")
        density, land, template = obm.load_parent_densities(
            pm_data, resolution, block_key
        )
        built = land & (obm.sum_parents(density, pmc.OBM_PARENTS) > 0)

        print("Loading GHSL height")
        height, n_imputed = obm.load_height(bd_data, resolution, block_key, built)

        print("Deriving OBM measures")
        # `check_features` asserts the six are mutually consistent - notably
        # `volume / density == height` - before any of them is written.
        features, n_rescaled = obm.derive_features(density, height)
        obm.check_features(features, height, land)

        n_built = int(built.sum())
        imputed_share = n_imputed / n_built if n_built else 0.0
        print(
            f"{block_key}: {n_built} built pixels, {n_rescaled} rescaled, "
            f"{n_imputed} height-imputed ({imputed_share:.2%})"
        )

        out_paths = {}
        for measure in self.built_version.measures:
            feature_name = f"{self.built_version.name}_{measure}"
            # Every measure carries the same nan mask, matching GHSL.
            raster = rt.RasterArray(
                np.where(land, features[measure], np.nan).astype(np.float32),
                transform=template.transform,
                crs=template.crs,
                no_data_value=np.nan,
            )
            pm_data.save_feature(
                raster,
                feature_name=feature_name,
                **self.feature_metadata.shared_kwargs,
            )
            out_paths[feature_name] = pm_data.feature_path(
                feature_name=feature_name,
                **self.feature_metadata.shared_kwargs,
            )
        return out_paths

    def generate_geospatial_averages(
        self,
        features: list[str],
        feature_average_radii: list[int],
        pm_data: PopulationModelData,
    ) -> dict[str, Path]:
        """Honour the version's opt-out.

        `geospatial_average_features` averages a fixed six measures across every
        registered version. OBM produces two of them, so running it would fail
        on the first missing raster rather than skip.
        """
        if not self.built_version.geospatial_averages:
            print(f"Geospatial averages disabled for {self.built_version.name}.")
            return {}
        return super().generate_geospatial_averages(
            features, feature_average_radii, pm_data
        )


class InterpolateStrategy(ProcessStrategy):
    def generate_measures(
        self, bd_data: BuildingDensityData, pm_data: PopulationModelData
    ) -> dict[str, Path]:
        out_paths = {}
        for measure in self.built_version.measures:
            out_measure = f"{self.built_version.name}_{measure}"
            tp_start, tp_end, w = get_time_points_and_weight(
                built_version=self.built_version,
                time_point=self.feature_metadata.time_point,
            )
            print(f"Interpolating {measure} for {self.built_version.name}.")
            print(
                f"Start: {tp_start}, End: {tp_end}, Weight: {w}, Time Point: {self.feature_metadata.time_point}"
            )
            print("Loading tiles")
            tile_start = bd_data.load_tile(
                resolution=self.feature_metadata.resolution,
                provider=self.built_version.name,
                block_key=self.feature_metadata.block_key,
                time_point=tp_start,
                measure=measure,
            )
            tile_end = bd_data.load_tile(
                resolution=self.feature_metadata.resolution,
                provider=self.built_version.name,
                block_key=self.feature_metadata.block_key,
                time_point=tp_end,
                measure=measure,
            )
            print("Interpolating and saving")
            built_measure = tile_start * w + tile_end * (1 - w)
            built_measure = utils.suppress_noise(built_measure)
            pm_data.save_feature(
                built_measure,
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
            out_paths[out_measure] = pm_data.feature_path(
                feature_name=out_measure,
                **self.feature_metadata.shared_kwargs,
            )
        return out_paths

    def link_features(
        self,
        feature_paths: dict[str, Path],  # noqa: ARG002
        fill_time_points: list[str],
        feature_metadata: FeatureMetadata,  # noqa: ARG002
        pm_data: PopulationModelData,  # noqa: ARG002
    ) -> None:
        assert fill_time_points == []  # noqa: S101
        print(
            f"Skipping {self.built_version.name} for {self.feature_metadata.time_point}."
        )


# Strategy classes a BuiltVersion may name. Keyed by class name so the
# registration in constants.py stays a plain string and does not have to import
# from here.
STRATEGIES: dict[str, type[ProcessStrategy]] = {
    "ObmStrategy": ObmStrategy,
}

# Versions that write every measure they have, so nothing is derived from them.
# GHSL ships its full set; OBM's ObmStrategy writes all six itself.
NO_DERIVED_MEASURES = frozenset({"ghsl_r2023a", pmc.OBM_PROVIDER})

HEIGHT_MIN = 2.4384  # 8ft


# How each version derives its measures from the three source rasters it names.
# `measures` is optional and selects which of `out_ops` to emit; the default is
# all of them.
#
# Only versions that *borrow* a residential fraction appear here. GHSL ships
# every measure it has, and OBM writes its own through ObmStrategy - see
# NO_DERIVED_MEASURES.
FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "microsoft_v6": {
        "density": "microsoft_v6_density",
        "height": "ghsl_r2023a_height",
        "p_residential": "ghsl_r2023a_proportion_residential",
    },
    "microsoft_v7": {
        "density": "microsoft_v7_density",
        "height": "microsoft_v7_height",
        "p_residential": "ghsl_r2023a_proportion_residential",
    },
    "microsoft_v7_1": {
        "density": "microsoft_v7_1_density",
        "height": "microsoft_v7_1_height",
        "p_residential": "ghsl_r2023a_proportion_residential",
    },
    "microsoft_v7_e101": {
        "density": "microsoft_v7_e101_density",
        "height": "microsoft_v7_e101_height",
        "p_residential": "ghsl_r2023a_proportion_residential",
    },
    "microsoft_v8": {
        "density": "microsoft_v8_density",
        "height": "microsoft_v8_height",
        "p_residential": "ghsl_r2023a_proportion_residential",
        # The shipped v8 carries only these. residential_density and the
        # nonresidential_* measures were never built for it, and nothing
        # downstream reads them, so emitting the full six would put four
        # unused rasters per block-epoch into the tree.
        "measures": (
            "density",
            "volume",
            "residential_volume",
            "proportion_residential",
        ),
    },

}


def _generate_provider_derived_measures(
    pm_data: PopulationModelData,
    feature_metadata: FeatureMetadata,
    built_version_name: str,
) -> dict[str, Path]:
    feature_dict = FEATURE_SPECS[built_version_name]
    density = pm_data.load_feature(
        feature_name=feature_dict["density"],
        **feature_metadata.shared_kwargs,
    )
    density_arr = density._ndarray  # noqa: SLF001
    height_arr = pm_data.load_feature(  # noqa: SLF001
        feature_name=feature_dict["height"],
        **feature_metadata.shared_kwargs,
    )._ndarray
    # Resolve p through the shared splice rather than reading the raster raw.
    # GHSL writes 0.0 where it sees no building; multiplying by that directly
    # zeroes the residential volume on pixels the shipped products keep fully
    # residential - 13.9% of v8's built pixels on a test block.
    p_provider = feature_dict["p_residential"].removesuffix(
        "_proportion_residential"
    )
    p_residential_arr, _ = splice_p([(
        pm_data.load_feature(
            feature_name=feature_dict["p_residential"],
            **feature_metadata.shared_kwargs,
        ).to_numpy(),
        pm_data.load_feature(
            feature_name=f"{p_provider}_density",
            **feature_metadata.shared_kwargs,
        ).to_numpy(),
    )])

    # Since we're crosswalking, ensure we have height wherever
    # there is density, even if GHSL doesn't think there is density.
    height_min = HEIGHT_MIN
    if (height_arr > 0).any():
        height_min = float(np.nanmin(height_arr[height_arr > 0]))
    density_threshold = 0.01
    density_is_positive = density_arr >= density_threshold
    height_is_zero = height_arr == 0
    height_arr[density_is_positive & height_is_zero] = height_min

    out_ops = {
        "density": lambda d, _, __: d,
        "residential_density": lambda d, _, p: d * p,
        "nonresidential_density": lambda d, _, p: d * (1 - p),
        "volume": lambda d, h, _: h * d,
        "residential_volume": lambda d, h, p: h * d * p,
        "nonresidential_volume": lambda d, h, p: h * d * (1 - p),
    }
    out_ops["proportion_residential"] = lambda _, __, p: p
    measures = feature_dict.get("measures", tuple(out_ops))
    out_ops = {m: out_ops[m] for m in measures}

    for measure, op in out_ops.items():
        # Cast explicitly: the shipped rasters are float32, and the writer's
        # PREDICTOR=2 rejects 64-bit samples. p comes back from splice_p as
        # float64, which would otherwise promote every product.
        out = rt.RasterArray(
            data=op(density_arr, height_arr, p_residential_arr).astype(np.float32),  # type: ignore[no-untyped-call]
            transform=density.transform,
            crs=density.crs,
            no_data_value=np.nan,
        )
        pm_data.save_feature(
            out,
            feature_name=f"{built_version_name}_{measure}",
            **feature_metadata.shared_kwargs,
        )

    out_paths = {
        f"{built_version_name}_{m}": pm_data.feature_path(
            feature_name=f"{built_version_name}_{m}",
            **feature_metadata.shared_kwargs,
        )
        for m in out_ops
    }
    return out_paths


def get_time_points_and_weight(
    built_version: pmc.BuiltVersion,
    time_point: str,
) -> tuple[str, str, float]:
    year, quarter = time_point.split("q")
    time_point_float = float(year) + (float(quarter) - 1) / 4
    bv_ftps = built_version.time_points_float

    tp_start, tp_end, w = "", "", 0.0
    for i, (t_start, t_end) in enumerate(zip(bv_ftps[:-1], bv_ftps[1:], strict=False)):
        if t_start <= time_point_float <= t_end:
            tp_start = built_version.time_points[i]
            tp_end = built_version.time_points[i + 1]
            w = (t_end - time_point_float) / (t_end - t_start)
    return tp_start, tp_end, w


def get_processing_strategy(
    built_version: pmc.BuiltVersion,
    feature_metadata: FeatureMetadata,
) -> tuple[ProcessingStrategy, list[str]]:
    """Determine the processing strategy for a given time point.

    This method is used to flexibly determine a processing strategy for a given time
    point. The strategy is determined based on the time point and the built version.
    The strategy is one of the following:

        - "process": The time point is in the built version. We process normally. If
            the time point is also terminal, we will also fill in the time points
            before and after the time point.
        - "skip": The time point is beyond the terminal time points of the built
            version. In this case, we will skip the time point as it will be extrapolated
            when we process the terminal time point.
        - "interpolate": The time point is between two time points in the built version.
            In this case, we will interpolate the time point and process it.

    """
    time_point = feature_metadata.time_point
    bv_tps = built_version.time_points
    bv_ftps = built_version.time_points_float
    first, last = bv_tps[0], bv_tps[-1]

    year, quarter = time_point.split("q")
    time_point_float = float(year) + (float(quarter) - 1) / 4

    # A version may name the class that builds its own measures; everything
    # else about the dispatch is unchanged. Only the "process" branch honours
    # it - skipping and interpolating are the same work whatever the source.
    process_cls: type[ProcessStrategy] = ProcessStrategy
    if built_version.strategy is not None:
        process_cls = STRATEGIES[built_version.strategy]

    strategy: ProcessingStrategy
    fill_time_points = []
    if time_point in bv_tps:
        # If the time point is in the built version, we process
        strategy = process_cls(
            built_version=built_version, feature_metadata=feature_metadata
        )
        # If the time point is also terminal, we extrapolate as well
        # A time point can be first, last, or both (i.e. if the version has a
        # single time point)
        if time_point == first:
            fill_time_points.extend(
                pmc.ALL_TIME_POINTS[: pmc.ALL_TIME_POINTS.index(first)]
            )
        if time_point == last:
            fill_time_points.extend(
                pmc.ALL_TIME_POINTS[pmc.ALL_TIME_POINTS.index(last) + 1 :]
            )
    elif time_point_float < bv_ftps[0] or time_point_float > bv_ftps[-1]:
        # If the time point is before the first or after the last time point,
        # it will be extrapolated when we process the terminal time point,
        # so we don't need to do anything here.
        strategy = SkipStrategy(
            built_version=built_version, feature_metadata=feature_metadata
        )
    else:
        # The time point is between two time points, so we need to interpolate.
        strategy = InterpolateStrategy(
            built_version=built_version, feature_metadata=feature_metadata
        )

    return strategy, fill_time_points


def mosaic_tile(
    measure: str,
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
) -> rt.RasterArray:
    tiles = []
    for bounds in feature_metadata.block_bounds.values():
        try:
            tile = pm_data.load_feature(
                feature_name=measure,
                **feature_metadata.shared_kwargs,
                subset_bounds=bounds,
            )
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
            tiles.append(tile)
        except ValueError:
            # This is kind of a hack, but there's not a clean way to fix it easily.
            # The issue is that the resolution of the tiles do not exactly line up
            # to the bounds of the world as defined by the CRS. The southernmost
            # have one fewer row of pixels as a whole row would extend past the
            # southern edge of the world, causing reprojection issues. The problem
            # is that we read the tile with bounds, and those bounds cause the underlying
            # rasterio to fill in that missing row. Here we just remove the last row
            # of pixels and reproject the tile.
            tile._ndarray = tile._ndarray[:-1].copy()  # noqa: SLF001
            tile = tile.reproject(
                dst_crs=feature_metadata.working_crs,
                dst_resolution=float(feature_metadata.resolution),
                resampling="average",
            )
            tiles.append(tile)

    buffered_measure = utils.suppress_noise(rt.merge(tiles))
    return buffered_measure
