"""Build Microsoft v8 with OBM supplying the residential split.

Microsoft v8 supplies its own density and height. The only thing it borrows is a
residential fraction `p`, which today comes from GHSL:

    p = ghsl_proportion_residential   where ghsl_density > 0
        1.0                           elsewhere
    v8_residential_volume = v8_volume x p

That reconstruction reproduces the shipped raster exactly, which is what makes a
source swap attributable. This module generalises it to a priority list of
sources - OBM first, GHSL second, the fully-residential fallback last - so the
layer stays global while every pixel that differs from `microsoft_v8` differs
because OBM had an opinion there.

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
    is only interpretable alongside `sources` - which lists the providers that
    were actually present, not the ones that were asked for.
    """

    p: NDArray[np.float64]
    owner: NDArray[np.int8]
    sources: list[str]
    counts: dict[str, int]

    def owned_by(self, provider: str) -> NDArray[np.bool_]:
        """Mask of pixels whose fraction came from `provider`."""
        if provider not in self.sources:
            return np.zeros(self.owner.shape, dtype=bool)
        mask: NDArray[np.bool_] = self.owner == self.sources.index(provider) + 1
        return mask

    @property
    def unsourced(self) -> NDArray[np.bool_]:
        """Mask of pixels no source saw, which take the fallback."""
        mask: NDArray[np.bool_] = self.owner == 0
        return mask


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
    """
    template = pm_data.load_feature(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{pmc.MSFT_V8_SOURCE_PROVIDER}_volume",
    )
    # Lowest priority first, so the preferred source is applied last and wins.
    layers, present = [], []
    for provider in reversed(sources):
        if not pm_data.feature_exists(
            resolution=resolution, block_key=block_key, time_point=time_point,
            feature_name=f"{provider}_proportion_residential",
        ):
            continue
        def load(measure: str, _p: str = provider) -> NDArray[Any]:
            return pm_data.load_feature(
                resolution=resolution, block_key=block_key, time_point=time_point,
                feature_name=f"{_p}_{measure}",
            ).to_numpy()

        layers.append((load("proportion_residential"), load("density")))
        present.append(provider)

    if not layers:
        shape = template.to_numpy().shape
        p = np.full(shape, pmc.MSFT_V8_OBM_P_FALLBACK, dtype=np.float64)
        owner = np.zeros(shape, dtype=np.int8)
    else:
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
    if not pm_data.feature_exists(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{src}_volume",
    ):
        print(f"No {src} volume for {block_key} at {time_point}; skipping.")
        return

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


def check_reproduces_source(
    pm_data: PopulationModelData,
    resolution: str,
    block_key: str,
    time_point: str,
    tolerance: float = 1e-6,
) -> float:
    """Verify the GHSL-only path reproduces the shipped microsoft_v8 raster.

    This is the precondition for the whole comparison: if feeding the same
    source GHSL uses does not return the shipped layer, then any later
    difference could be our arithmetic rather than the source swap.
    """
    src = pmc.MSFT_V8_SOURCE_PROVIDER
    p = resolve_p(pm_data, resolution, block_key, time_point, ("ghsl_r2023a",)).p
    vol = pm_data.load_feature(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{src}_volume").to_numpy()
    shipped = pm_data.load_feature(
        resolution=resolution, block_key=block_key, time_point=time_point,
        feature_name=f"{src}_residential_volume").to_numpy()
    built = np.isfinite(vol) & (vol > 0)
    if not built.any():
        return 0.0
    err = float(np.abs((vol * p)[built] - shipped[built]).max())
    if err > tolerance:
        msg = (f"GHSL-only reconstruction does not reproduce {src} on {block_key} "
               f"at {time_point}: max|diff| = {err}")
        raise ValueError(msg)
    return err
