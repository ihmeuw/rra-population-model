"""Build the Open Building Map features from the OBM covariate rasters.

The covariate ships eight parent building types x two measures per block. This
module collapses those into the six measures the model consumes, mirroring the
shape of `ghsl_r2023a` and `microsoft_v8`:

    {provider}_density
    {provider}_height
    {provider}_volume
    {provider}_residential_volume
    {provider}_proportion_residential
    {provider}_p_observed

OBM is a static snapshot, so the real files are written once into a canonical
time point and every other time point links to them.
"""

from pathlib import Path

import numpy as np
import rasterra as rt
from numpy.typing import NDArray

from rra_population_model import constants as pmc
from rra_population_model.data import BuildingDensityData, PopulationModelData


def load_parent_densities(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_], rt.RasterArray]:
    """Read the eight parent density rasters and the shared land mask.

    All eight share one nodata mask, so any of them defines the land domain.
    """
    template = pm_data.load_open_building_map_covariate(
        resolution=resolution,
        block_key=block_key,
        parent_building_type=pmc.OBM_PARENTS[0],
        measure="density",
    )
    land = ~np.isnan(template.to_numpy())

    density = {}
    for parent in pmc.OBM_PARENTS:
        if parent == pmc.OBM_PARENTS[0]:
            raster = template
        else:
            raster = pm_data.load_open_building_map_covariate(
                resolution=resolution,
                block_key=block_key,
                parent_building_type=parent,
                measure="density",
            )
        density[parent] = np.nan_to_num(raster.to_numpy()).astype(np.float64)

    return density, land, template


def load_height(
    bd_data: BuildingDensityData,
    resolution: str,
    block_key: str,
    built: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], int]:
    """Read GHSL ANBH and impute one storey where it sees no height.

    OBM's own height column is not used: OBM derives its heights from GHSL and
    records them as GEM taxonomy storey *ranges*, so reading GHSL directly follows
    OBM's own convention and avoids guessing a range midpoint and a
    metres-per-storey factor.

    The imputation is confined to pixels where OBM sees a footprint. Elsewhere it
    would not change any derived feature - volume is zero wherever density is -
    but it would fill the shipped height raster with spurious 2.5s and make the
    imputation share unreportable. Because it is confined this way, the imputation
    mask is exactly `ghsl_r2023a_height == 0` intersected with a positive OBM
    density, which is why we do not ship it as a sixth raster.
    """
    ghsl_height = bd_data.load_tile(
        resolution=resolution,
        provider="ghsl_r2023a",
        block_key=block_key,
        time_point=pmc.OBM_GHSL_TIME_POINT,
        measure="height",
    )
    height = np.nan_to_num(ghsl_height.to_numpy()).astype(np.float64)
    imputed = built & (height == 0)
    height = np.where(imputed, pmc.OBM_IMPUTED_HEIGHT, height)
    return height, int(imputed.sum())


def sum_parents(
    density: dict[str, NDArray[np.float64]],
    parents: list[str],
) -> NDArray[np.float64]:
    """Accumulate parent layers in place.

    Stacking the eight 8192^2 layers to sum along an axis would cost ~4 GB for no
    benefit, so accumulate instead.
    """
    total = np.zeros_like(density[parents[0]])
    for parent in parents:
        total += density[parent]
    return total


def derive_features(
    density: dict[str, NDArray[np.float64]],
    height: NDArray[np.float64],
) -> tuple[dict[str, NDArray[np.float64]], int]:
    """Rescale the double-counted pixels, then derive the five features.

    Order matters: rescale before deriving anything, or every derived raster
    inherits the double count.

    A pixel's parent densities can sum above 1 because OSM building relations keep
    an enclosing building *and* its constituent parts as separate rows, each free
    to carry a different occupancy code, so the same ground lands in several parent
    layers. We rescale by 1/total rather than clipping: clipping would break both
    `volume / density == height` and `residential + nonres == total`, while
    rescaling removes the double count and preserves the use mix, leaving
    `proportion_residential` unchanged. It is a rare artefact - 0.0089% of built
    pixels, 0.022% of built area - but it is not a rasterization defect.
    """
    total_raw = sum_parents(density, pmc.OBM_PARENTS)
    # The tolerance matters for the reported count, not the arithmetic. Layers are
    # stored as float32 and accumulated here in float64, so pixels summing to
    # exactly 1.0 can land a rounding step above it. Without the tolerance they
    # count as double-counted and get scaled by 1 - 1e-16, which is a no-op that
    # inflates the diagnostic; with it, the count matches the covariate analysis.
    rescaled = total_raw > 1 + pmc.OBM_RESCALE_TOLERANCE
    scale = np.where(rescaled, 1 / np.where(rescaled, total_raw, 1.0), 1.0)

    total = total_raw * scale
    nonres = sum_parents(density, pmc.OBM_NONRESIDENTIAL_PARENTS) * scale

    # GHSL writes 0.0, not NaN, where there is no building; match it exactly.
    proportion_residential = np.where(
        total > 0, 1 - nonres / np.where(total > 0, total, 1.0), 0.0
    )
    volume = height * total

    features = {
        "density": total,
        "volume": volume,
        "residential_volume": volume * proportion_residential,
        "proportion_residential": proportion_residential,
        "height": height,
        # Where OBM actually saw a building. Identical to `density > 0`, and
        # shipped anyway: downstream this is the line between a measured
        # residential fraction and an imputed one, and `microsoft_v8_obm` has no
        # way to express it - OBM's own p = 1 and the no-source fallback write
        # the same number.
        pmc.OBM_OBSERVED_MEASURE: (total > 0).astype(np.float64),
    }
    return features, int(rescaled.sum())


def check_features(
    features: dict[str, NDArray[np.float64]],
    height: NDArray[np.float64],
    land: NDArray[np.bool_],
) -> None:
    """Assert the invariants the features are supposed to carry."""
    if not land.any():
        # Some blocks are entirely nodata - open ocean, for instance. There is
        # nothing to check, and every reduction below is empty. The features are
        # still written, all-NaN, so the block's shape matches its neighbours'.
        return

    density = features["density"]
    proportion_residential = features["proportion_residential"]

    tolerance = 1e-5
    max_density = density[land].max()
    if max_density > 1 + tolerance:
        msg = f"density exceeds 1 after rescaling: max {max_density}"
        raise ValueError(msg)

    p_on_land = proportion_residential[land]
    if p_on_land.min() < -tolerance or p_on_land.max() > 1 + tolerance:
        msg = (
            "proportion_residential outside [0, 1]: "
            f"[{p_on_land.min()}, {p_on_land.max()}]"
        )
        raise ValueError(msg)

    built = land & (density > 0)
    if built.any():
        implied_height = features["volume"][built] / density[built]
        max_diff = np.abs(implied_height - height[built]).max()
        if max_diff > tolerance:
            msg = f"volume / density != height: max difference {max_diff}"
            raise ValueError(msg)


def process_obm(
    pm_data: PopulationModelData,
    bd_data: BuildingDensityData,
    resolution: str,
    block_key: str,
    time_point: str,
) -> None:
    """Build and link the OBM features for one block.

    Takes plain arguments rather than a FeatureMetadata: everything OBM needs
    comes from the covariate rasters themselves, so requiring the full metadata
    would mean loading the modelling frame and a template tile per block for
    nothing.
    """
    if time_point != pmc.OBM_TIME_POINT:
        # OBM is a static snapshot. Every other time point is a symlink, created
        # when the canonical time point runs.
        return

    covariate_root = pm_data.open_building_map_covariates / f"{resolution}m" / block_key
    if not covariate_root.exists():
        print(f"No OBM coverage for block {block_key}; skipping.")
        return

    print(f"Loading OBM parent densities for {block_key}")
    density, land, template = load_parent_densities(pm_data, resolution, block_key)
    built = land & (sum_parents(density, pmc.OBM_PARENTS) > 0)

    print("Loading GHSL height")
    height, n_imputed = load_height(bd_data, resolution, block_key, built)

    print("Deriving features")
    features, n_rescaled = derive_features(density, height)
    check_features(features, height, land)

    # The two numbers the covariate analysis quotes, reproducible from this run.
    n_built = int(built.sum())
    imputed_share = n_imputed / n_built if n_built else 0.0
    print(
        f"{block_key}: {n_built} built pixels, {n_rescaled} rescaled, "
        f"{n_imputed} height-imputed ({imputed_share:.2%})"
    )

    # Every feature carries the same nan mask, matching GHSL.
    shared_kwargs = {
        "resolution": resolution,
        "block_key": block_key,
        "time_point": time_point,
    }
    feature_paths: dict[str, Path] = {}
    for measure, array in features.items():
        feature_name = f"{pmc.OBM_PROVIDER}_{measure}"
        raster = rt.RasterArray(
            np.where(land, array, np.nan).astype(np.float32),
            transform=template.transform,
            crs=template.crs,
            no_data_value=np.nan,
        )
        pm_data.save_feature(raster, feature_name=feature_name, **shared_kwargs)
        feature_paths[feature_name] = pm_data.feature_path(
            feature_name=feature_name, **shared_kwargs
        )

    link_features(pm_data, feature_paths, resolution, block_key)


def link_features(
    pm_data: PopulationModelData,
    feature_paths: dict[str, Path],
    resolution: str,
    block_key: str,
) -> None:
    """Link every other time point at the canonical files.

    Each name is linked to its own target. The overture step links only the
    un-prefixed name, which left every `log_*` symlink pointing at the non-log
    file; iterating name/path pairs makes that mistake unrepresentable.
    """
    for feature_name, source_path in feature_paths.items():
        for time_point in pmc.ALL_TIME_POINTS:
            if time_point == pmc.OBM_TIME_POINT:
                continue
            pm_data.link_feature(
                source_path=source_path,
                feature_name=feature_name,
                time_point=time_point,
                block_key=block_key,
                resolution=resolution,
            )
