"""Build Microsoft v8 with OBM supplying the residential split.

Microsoft v8 supplies its own density and height. The only thing it borrows is a
residential fraction `p`, which today comes from GHSL:

    p = ghsl_proportion_residential   where ghsl_density > 0
        1.0                           elsewhere
    v8_residential_volume = v8_volume x p

That reconstruction reproduces `microsoft_v8` exactly - both go through
`splice_p` - which is what makes a source swap attributable. (v8 rasters built
before the splice existed differ from it before 2025q1; see
`_generate_provider_derived_measures`.) This module generalises it to a
priority list of sources - OBM first, GHSL second, the fully-residential
fallback last - so the layer stays global while every pixel that differs from
`microsoft_v8` differs because OBM had an opinion there.

Note the fallback is *not* the same as reading `proportion_residential` directly.
GHSL's raster stores 0.0 where it sees no building, so multiplying by it raw
would zero the residential volume on pixels that `microsoft_v8` keeps fully
residential - 13.9% of its built pixels on a test block. The fallback has to be
applied per source, on that source's own building mask.
"""


from typing import Any, NamedTuple

import numpy as np
import rasterra as rt
from numpy.typing import NDArray

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData


def splice_p(
    layers: list[tuple[NDArray[Any], NDArray[Any]]],
    fallback: float = pmc.MSFT_V8_OBM_P_FALLBACK,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Combine residential fractions from several sources, lowest priority first.

    `layers` is a list of (fraction, density) pairs. A source only speaks where
    its own density is positive; everything unclaimed keeps `fallback`. Applying
    the list in order means the last entry wins, so callers pass their preferred
    source last.

    This is the one place the fallback rule lives. Reading a
    `proportion_residential` raster directly is not equivalent: GHSL writes 0.0
    where it sees no building, so multiplying by it raw zeroes the residential
    volume on pixels the shipped products keep fully residential.
    """
    p = np.full(layers[0][0].shape, fallback, dtype=np.float64)
    owner = np.zeros(layers[0][0].shape, dtype=np.int8)
    for rank, (p_src, d_src) in enumerate(layers, start=1):
        seen = np.isfinite(d_src) & (d_src > 0)
        p = np.where(seen, np.nan_to_num(p_src).astype(np.float64), p)
        owner = np.where(seen, rank, owner)
    return p, owner


class ResolvedP(NamedTuple):
    """The spliced fraction, and the record of where each pixel came from.

    `owner` is 0 where no source spoke and `i + 1` where `sources[i]` did, so it
    is only interpretable alongside `sources`, the providers used, lowest
    priority first.
    """

    p: NDArray[np.float64]
    owner: NDArray[np.int8]
    sources: list[str]
    counts: dict[str, int]


def resolve_p(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
    time_point: str,
    sources: tuple[str, ...],
) -> ResolvedP:
    """Residential fraction from the first source that sees a building.

    Sources are applied in order, each overwriting the last, so the *last*
    source in `sources` is the lowest priority. Everything unclaimed keeps the
    fully-residential fallback.

    Returns the fraction, the per-pixel record of which source supplied it, and
    a per-source count - the last two being the only way to tell a measured
    fraction from an assumed one downstream.

    Every source must have been built at this time point. A missing one raises
    rather than being skipped: skipping is how the product came out GHSL-only,
    under the OBM name, when it was built before OBM existed. For the same
    reason an empty `sources` raises rather than returning the fallback
    everywhere.
    """
    if not sources:
        msg = (
            f"No residential-fraction sources for {block_key} at {time_point}; "
            f"p would be {pmc.MSFT_V8_OBM_P_FALLBACK} everywhere."
        )
        raise ValueError(msg)
    template = pm_data.load_feature(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{pmc.MSFT_V8_SOURCE_PROVIDER}_volume",
    )
    # Lowest priority first, so the preferred source is applied last and wins.
    layers, present = [], []
    for provider in reversed(sources):
        for measure in ("proportion_residential", "density"):
            if not pm_data.feature_exists(
                resolution=resolution, block_key=block_key, time_point=time_point,
                feature_name=f"{provider}_{measure}",
            ):
                command = (
                    "obm_features" if provider == pmc.OBM_PROVIDER else "features"
                )
                msg = (
                    f"No {provider}_{measure} for {block_key} at {time_point} "
                    f"({resolution}m). Run `pmrun model_prep {command}` first."
                )
                raise FileNotFoundError(msg)
        def load(measure: str, _p: str = provider) -> NDArray[Any]:
            return pm_data.load_feature(
                resolution=resolution, block_key=block_key, time_point=time_point,
                feature_name=f"{_p}_{measure}",
            ).to_numpy()

        layers.append((load("proportion_residential"), load("density")))
        present.append(provider)

    p, owner = splice_p(layers)

    # Count over the pixels v8 actually calls built. Counting over all land
    # would report ~84% fallback on a typical block simply because most land
    # holds no buildings, which says nothing about the layer.
    vol = template.to_numpy()
    built = np.isfinite(vol) & (vol > 0)
    counts = {"fallback": int((built & (owner == 0)).sum())}
    for i, provider in enumerate(present, start=1):
        counts[provider] = int((built & (owner == i)).sum())
    return ResolvedP(p=p, owner=owner, sources=present, counts=counts)


def process_msft_obm(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
    time_point: str,
    *,
    sources: tuple[str, ...] = pmc.MSFT_V8_OBM_P_SOURCES,
    provider: str = pmc.MSFT_V8_OBM_PROVIDER,
) -> None:
    """Build and link the Microsoft-v8-with-OBM layers for one block."""
    src = pmc.MSFT_V8_SOURCE_PROVIDER
    # Every modeling-frame block has Microsoft v8 - `features` raises on a
    # missing tile - so a missing volume means `features` has not run here.
    if not pm_data.feature_exists(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{src}_volume",
    ):
        msg = (
            f"No {src}_volume for {block_key} at {time_point} ({resolution}m). "
            "Run `pmrun model_prep features` first."
        )
        raise FileNotFoundError(msg)

    # OBM covers 2,902 of 4,797 blocks, so a missing OBM raster is expected on
    # the rest; everywhere else it means `obm_features` has not run, and
    # `resolve_p` raises. Coverage is the covariate's block directory existing -
    # the same test ObmStrategy uses to decide whether to build the block.
    if pmc.OBM_PROVIDER in sources:
        obm_root = pm_data.open_building_map_covariates / f"{resolution}m"
        if not obm_root.is_dir():
            msg = (
                f"No OBM covariate at {resolution}m ({obm_root}), so "
                f"{provider} would carry no OBM at all."
            )
            raise FileNotFoundError(msg)
        if not (obm_root / block_key).exists():
            print(f"No OBM coverage for {block_key}; p from the other sources.")
            sources = tuple(s for s in sources if s != pmc.OBM_PROVIDER)

    print(f"Resolving p for {block_key} at {time_point}")
    resolved = resolve_p(pm_data, resolution, block_key, time_point, sources)
    p, counts = resolved.p, resolved.counts
    total = sum(counts.values())
    if total:
        summary = "  ".join(f"{k}: {v / total:.1%}" for k, v in counts.items())
        print(f"  p source shares — {summary}")

    volume = pm_data.load_feature(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{src}_volume",
    )
    vol = volume.to_numpy()
    land = np.isfinite(vol)

    shared_kwargs = {
        "resolution": resolution, "block_key": block_key, "time_point": time_point,
    }
    # One product. The spliced fraction is not written: it is recoverable as
    # `residential_volume / microsoft_v8_volume` wherever the denominator is
    # non-zero, and undefined exactly where it has no effect. The observation
    # mask is not written either - it describes OBM, not the splice, so it is
    # built once by the OBM step rather than 24 times here.
    for measure, array in (
        ("residential_volume", vol * p),
    ):
        raster = rt.RasterArray(
            np.where(land, array, np.nan).astype(np.float32),
            transform=volume.transform,
            crs=volume.crs,
            no_data_value=np.nan,
        )
        pm_data.save_feature(raster, feature_name=f"{provider}_{measure}",
                             **shared_kwargs)

    # Microsoft's own measures are untouched by the swap, so link rather than
    # copy - it keeps the two providers provably identical on those layers.
    for measure in pmc.MSFT_V8_OBM_LINKED_MEASURES:
        source_path = pm_data.feature_path(feature_name=f"{src}_{measure}",
                                           **shared_kwargs)
        if source_path.exists():
            pm_data.link_feature(source_path=source_path,
                                 feature_name=f"{provider}_{measure}",
                                 **shared_kwargs)

    _fill_earlier_time_points(pm_data, resolution, block_key, time_point, provider)


def _fill_earlier_time_points(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
    time_point: str,
    provider: str,
) -> None:
    """Back-extrapolate to every time point before Microsoft v8 begins.

    This mirrors `microsoft_v8` exactly: it holds a real file at each of its own
    24 quarters and links every earlier time point at the *first* of them, rather
    than at a nearest neighbour. So the fill only fires when the first epoch is
    the one being built.
    """
    epochs = pmc.BUILT_VERSIONS[pmc.MSFT_V8_SOURCE_PROVIDER].time_points
    if time_point != epochs[0]:
        return
    earlier = pmc.ALL_TIME_POINTS[: pmc.ALL_TIME_POINTS.index(epochs[0])]
    if not earlier:
        return

    measures = (*pmc.MSFT_V8_OBM_LINKED_MEASURES, *pmc.MSFT_V8_OBM_DERIVED_MEASURES)
    print(f"  back-filling {len(earlier)} earlier time points from {time_point}")
    for measure in measures:
        source_path = pm_data.feature_path(
            resolution=resolution, block_key=block_key, time_point=time_point,
            feature_name=f"{provider}_{measure}",
        )
        if not source_path.exists():
            continue
        for earlier_tp in earlier:
            pm_data.link_feature(
                source_path=source_path,
                feature_name=f"{provider}_{measure}",
                resolution=resolution,
                block_key=block_key,
                time_point=earlier_tp,
            )
