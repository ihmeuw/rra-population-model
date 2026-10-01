"""Derive the Open Building Map measures from the OBM covariate rasters.

The covariate ships seven labelled parent building types, each split into a
`tagged` and an `inherited` layer, plus `unknown`, at two measures per block.
`load_parent_densities` recombines those into the eight parents according to the
built version's `obm_inherited_to_unknown`, and the rest of this module collapses
the parents into the six measures the model consumes, mirroring the shape of
`ghsl_r2023a` and `microsoft_v8`:

    {provider}_density
    {provider}_height
    {provider}_volume
    {provider}_residential_volume
    {provider}_proportion_residential
    {provider}_p_observed

Nothing here writes. `ObmStrategy` in `built.py` calls these functions and
handles writing and the symlink fan-out, so OBM goes through the same
built-version machinery as every other provider - which is also where the
static-snapshot layout comes from: a single-epoch version fills every other time
point by symlink automatically.

The `unknown` parent - a footprint OBM measured but could not identify - is
credited at GHSL's local residential rate rather than assumed to be housing
outright. See `load_ghsl_credit` and `derive_features` for why.
"""

import numpy as np
import rasterra as rt
from numpy.typing import NDArray

from rra_population_model import constants as pmc
from rra_population_model.data import BuildingDensityData, PopulationModelData
from rra_population_model.model_prep.features.msft_obm import splice_p


def load_parent_densities(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
    inherited_to_unknown: list[str] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_], rt.RasterArray]:
    """Read the label-source density rasters and recombine them into parents.

    The covariate writes each labelled parent as two layers, `{parent}_tagged`
    and `{parent}_inherited`, and `unknown` whole. Each layer is added into its
    parent, except that the inherited layer of every parent in
    `inherited_to_unknown` is added into `unknown` instead: those footprints
    lose their zone-inherited label and are credited like any other unlabelled
    footprint. With nothing named, every parent gets both halves back, which
    reproduces the unsplit covariate.

    "Reproduces" up to one thing. The unsplit covariate rasterized a parent in
    one pass, so overlapping footprints of that parent were unioned; split, a
    tagged and an inherited footprint of the same parent that overlap are
    counted once in each half. The covariate's own check measured this at 109
    of 4.8M built pixels on the Kigali block, all positive. Any pixel it pushes
    above 1 is caught by the rescale in `derive_features`.

    Layers are accumulated into the eight parent arrays as they are read rather
    than all held at once: fifteen 8192^2 float64 layers would double peak
    memory for no benefit. All layers share one nodata mask, so `unknown`, the
    one layer every version reads, defines the land domain.
    """
    move = set(inherited_to_unknown or ())

    def load(layer: str) -> rt.RasterArray:
        return pm_data.load_open_building_map_covariate(
            resolution=resolution,
            block_key=block_key,
            parent_building_type=layer,
            measure="density",
        )

    template = load("unknown")
    land = ~np.isnan(template.to_numpy())

    density = {
        parent: np.zeros(land.shape, dtype=np.float64) for parent in pmc.OBM_PARENTS
    }
    density["unknown"] += np.nan_to_num(template.to_numpy()).astype(np.float64)
    for parent in pmc.OBM_LABELLED_PARENTS:
        for source in pmc.OBM_LABEL_SOURCES:
            target = (
                "unknown" if source == "inherited" and parent in move else parent
            )
            layer = load(f"{parent}_{source}").to_numpy()
            density[target] += np.nan_to_num(layer).astype(np.float64)

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


def load_ghsl_credit(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The residential rate to credit OBM's `unknown` footprints at.

    An unlabelled footprint is not evidence of housing, but it is not evidence
    against it either, so it is credited at whatever GHSL says about the same
    ground instead of at 1.0.

    `splice_p` rather than the raster directly, for the reason its own docstring
    gives: GHSL writes 0.0 where it sees no building, so reading it raw would
    treat "GHSL is blind here" as "nothing here is residential" and strip the
    unknown volume out entirely. `splice_p` confines GHSL to the pixels it can
    actually see and leaves the rest at `MSFT_V8_OBM_P_FALLBACK`, which is the
    same fallback every other provider in this package gets. Keeping that rule
    in one place matters more than the alternative of a local mean: it is under
    1% of v8 volume either way, and a second, differently-behaved fallback would
    be a worse thing to own.

    Read at `OBM_GHSL_TIME_POINT`, matching `load_height` - OBM is a static
    snapshot and borrows a single GHSL epoch for everything.
    """

    def load(measure: str) -> NDArray[np.float64]:
        raster = pm_data.load_feature(
            resolution=resolution,
            block_key=block_key,
            feature_name=f"ghsl_r2023a_{measure}",
            time_point=pmc.OBM_GHSL_TIME_POINT,
        )
        return np.asarray(raster.to_numpy(), dtype=np.float64)

    credit, owner = splice_p([(load("proportion_residential"), load("density"))])
    return credit, np.asarray(owner > 0, dtype=np.bool_)


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
    ghsl_credit: NDArray[np.float64],
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

    # `unknown` stays out of OBM_NONRESIDENTIAL_PARENTS and stays in `total`;
    # what changes is that it is no longer credited as housing outright. The
    # complement of GHSL's local rate joins the non-residential sum, leaving the
    # residential fraction as residential_mu plus a GHSL-credited share of
    # unknown, over the same total.
    #
    # The split has to be fractional, which is why it lives here rather than in
    # the parent list: naming `unknown` non-residential would assert the
    # opposite extreme, and `sum_parents` can only take a whole layer.
    # `* scale` because `nonres` is already rescaled, and dropping it would
    # break `residential + nonres == total` on the double-counted pixels.
    unk_nonres = (1.0 - ghsl_credit) * density["unknown"] * scale
    nonres = nonres + unk_nonres

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
