import itertools
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pyproj
from pydantic import BaseModel, model_validator

RRA_ROOT = Path("/mnt/team/rapidresponse/")
RRA_CREDENTIALS_ROOT = RRA_ROOT / "priv" / "shared" / "credentials"
RRA_BINARIES_ROOT = RRA_ROOT / "priv" / "shared" / "bin"
GEOSPATIAL_COVARIATES_ROOT = Path("/snfs1/WORK/11_geospatial/01_covariates")
BUILDING_DENSITY_ROOT = RRA_ROOT / "pub" / "building-density"
POPULATION_DATA_ROOT = RRA_ROOT / "pub" / "population" / "data" / "02-processed-data"
MODEL_ROOT = RRA_ROOT / "pub" / "population-model"
POPULATION_COVARIATE_ROOT = (
    RRA_ROOT / "pub" / "population" / "data" / "02-processed-data" / "covariates"
)


class RESOLUTIONS(StrEnum):
    r40 = "40"
    r100 = "100"

    @classmethod
    def to_list(cls) -> list[str]:
        return [r.value for r in cls]


class BuiltVersion(BaseModel):
    provider: Literal["ghsl", "microsoft"]
    version: Literal["v6", "v7", "v7_1", "v8", "r2023a"]
    time_points: list[str]
    measures: list[str]

    @property
    def name(self) -> str:
        return f"{self.provider}_{self.version}"

    @property
    def time_points_float(self) -> list[float]:
        """Convert time points to floats."""
        out = []
        for tp in self.time_points:
            year, quarter = tp.split("q")
            out.append(float(year) + (float(quarter) - 1) / 4)
        return out


BUILT_VERSIONS = {
    "ghsl_r2023a": BuiltVersion(
        provider="ghsl",
        version="r2023a",
        time_points=[f"{y}q1" for y in range(1975, 2030, 5)],
        measures=[
            "height",
            "proportion_residential",
            "density",
            "residential_density",
            "nonresidential_density",
            "volume",
            "residential_volume",
            "nonresidential_volume",
        ],
    ),
    "microsoft_v6": BuiltVersion(
        provider="microsoft",
        version="v6",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2024), range(1, 5))
        ][1:],
        measures=["density"],
    ),
    "microsoft_v7": BuiltVersion(
        provider="microsoft",
        version="v7",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2024), range(1, 5))
        ][1:],
        measures=[
            "density",
            "height",
        ],
    ),
    "microsoft_v7_1": BuiltVersion(
        provider="microsoft",
        version="v7_1",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2025), range(1, 5))
        ][1:-2],
        measures=[
            "density",
            "height",
        ],
    ),
    # Built by newer code than this branch, but its tiles and features are on
    # disk for all 24 quarters. Registering it here is what makes ALL_TIME_POINTS
    # cover the feature directories that actually exist; without it the list
    # stops at 2025q1 and every step silently skips the six newest time points.
    # Note it is deliberately absent from the runner's BUILT_VERSIONS list, so
    # nothing here tries to rebuild it - `_generate_microsoft_derived_measures`
    # has no v8 entry and would raise if it were added there.
    "microsoft_v8": BuiltVersion(
        provider="microsoft",
        version="v8",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2027), range(1, 5))
        ][1:-3],
        measures=[
            "density",
            "height",
        ],
    ),

}

DENOMINATORS = []
for built_version in BUILT_VERSIONS.values():
    for denominator in [
        "density",
        "volume",
        "residential_density",
        "residential_volume",
    ]:
        DENOMINATORS.append(f"{built_version.name}_{denominator}")  # noqa: PERF401


# Open Building Map is a static snapshot, so it ships as a single vintage.
#
# `2025-04-04` is the last-modified date GFZ stamps on the `_data` folder of
# publication 2025-002 (doi:10.5880/GFZ.LKUT.2025.002), which is the only thing
# distinguishing one download from another - the package carries no version
# field. It pins *which upload we pulled*, not how current the data is: the
# footprints themselves carry OSM timestamps of 2024-09-26.
OBM_VERSION = "2025-04-04"
# The provider is that date without separators. Hyphens are out because feature
# names are addressed as `{provider}_{measure}`; extra underscores would make
# the provider/measure boundary unreadable, and dots break `Path.suffixes`.
OBM_PROVIDER = "obm_20250404"
# Real feature files are written here and every other time point links to them.
# This is an *epoch directory*, so it has to be one of ALL_TIME_POINTS - it
# cannot follow OBM_PROVIDER. 2025q2 is the quarter the snapshot falls in.
OBM_TIME_POINT = "2025q2"
# Where the heights come from. This is GHSL's epoch, not ours, and does not move
# with OBM_TIME_POINT.
OBM_GHSL_TIME_POINT = "2025q1"

# The eight parent building types the covariate rasterizes, in the order the
# covariate writes them.
OBM_PARENTS = [
    "residential_mu",
    "commercial",
    "industrial",
    "agriculture",
    "government",
    "education",
    "assembly",
    "unknown",
]
# `unknown` is deliberately excluded: it is treated as residential, because
# unlabelled footprints are dimensionally indistinguishable from labelled
# housing. `proportion_residential` is therefore 1 - nonres/total, which keeps
# `unknown` in the denominator but out of the numerator.
OBM_NONRESIDENTIAL_PARENTS = [
    p for p in OBM_PARENTS if p not in ("residential_mu", "unknown")
]

# GHSL's ANBH is continuous metres with no concept of a storey, but it has a hard
# empirical floor: the minimum non-zero value is ~2.486 m with the low-rise mass
# at ~2.50 m. We impute one storey wherever OBM sees a footprint and GHSL sees no
# height. This agrees with HEIGHT_MIN (2.4384 m, 8 ft) to within 5 cm.
OBM_IMPUTED_HEIGHT = 2.5

# The observation mask: 1 where OBM sees a building, 0 elsewhere on land.
#
# It lives with the OBM build rather than the v8 product because it describes
# OBM, not the splice - a single static snapshot, written once and linked like
# every other OBM measure, instead of 24 near-identical copies per block.
#
# Note it is exactly `{provider}_density > 0`, so it carries no information the
# density raster does not. It is shipped as an explicit boolean because the
# distinction between "measured" and "imputed" is the one consumers most need
# and most easily get wrong, and a named mask is harder to misread than a
# threshold someone has to know to apply.
OBM_OBSERVED_MEASURE = "p_observed"

# Pixels whose parent densities sum above this are rescaled by 1/total. See
# derive_features for why the tolerance is not simply zero.
OBM_RESCALE_TOLERANCE = 1e-6

# ---------------------------------------------------------------------------
# Microsoft v8, with OBM supplying the residential split instead of GHSL.
#
# Microsoft supplies its own density and height; the only thing it borrows is a
# residential fraction. `microsoft_v8` takes that from GHSL. This provider takes
# it from OBM where OBM has an opinion and falls back to GHSL where it does not,
# so the layer stays global and every pixel that differs from `microsoft_v8`
# differs because OBM said something - not because it said nothing.
# The vintage is carried in the name: this product is only meaningful relative
# to the OBM snapshot that supplied its residential split, and a future snapshot
# should not silently overwrite it.
MSFT_V8_OBM_PROVIDER = f"microsoft_v8_obm_{OBM_VERSION.replace('-', '')}"
MSFT_V8_SOURCE_PROVIDER = "microsoft_v8"
# Priority order for the residential fraction. First source with a building in
# the pixel wins; if none has one, the fraction falls back to fully residential,
# which is what `microsoft_v8` does today.
MSFT_V8_OBM_P_SOURCES = (OBM_PROVIDER, "ghsl_r2023a")
MSFT_V8_OBM_P_FALLBACK = 1.0
# Measures taken unchanged from microsoft_v8 - Microsoft's own density, height
# and volume, which the residential-source swap cannot touch.
#
# Deliberately empty: aliasing them under the `microsoft_v8_obm_` prefix would
# double the layer's path count (1.0M -> 2.0M) to publish files byte-identical
# to ones already on disk. Nothing reads them - the model loads a denominator as
# a single named raster, and `microsoft_v8_obm` is not in BUILT_VERSIONS, so it
# never reaches DENOMINATORS. Anyone wanting v8's density reads
# `microsoft_v8_density`, which is its canonical name.
#
# Re-populating this tuple is all that is needed to publish the aliases; they
# are symlinks, so a rebuild is seconds per block. Note that registering the
# provider as a denominator would *also* require `residential_density`, which
# nothing currently produces.
MSFT_V8_OBM_LINKED_MEASURES: tuple[str, ...] = ()
# Binary surface mask splitting the built footprint on the one distinction that
# matters downstream: was the residential fraction *observed* by OBM, or
# *imputed*? 1 where OBM supplied it, 0 where GHSL filled in or where no source
# saw anything and p = 1 was asserted.
#
# The mask is needed because `proportion_residential` cannot carry this. OBM's
# own p = 1 and the no-source fallback are the same number, so the raster alone
# gives a consumer no way to tell a measurement from an assumption - on one test
# block 98% of OBM's pixels sit at p = 1 already.
#
# GHSL's fill and the no-source fallback are deliberately on the same side of
# the split. A consumer who needs them apart can recover the fallback by
# intersecting the built footprint with the complement of this mask and the
# complement of GHSL's own density mask.
# Measures written as real rasters rather than linked from microsoft_v8.
#
# Only the one. `proportion_residential` was dropped because it is recoverable
# exactly wherever it means anything:
#     p = microsoft_v8_obm_..._residential_volume / microsoft_v8_volume
# and undefined where the denominator is zero, which is precisely where p has no
# effect. Writing it would cost another 115,128 rasters to store a quotient.
#
# The observation mask moved to the OBM build - see OBM_OBSERVED_MEASURE. It
# describes a static snapshot, so one raster serves all 24 epochs.
MSFT_V8_OBM_DERIVED_MEASURES = ("residential_volume",)

FEATURE_AVERAGE_RADII = [
    100,
    500,
    1000,
    2500,
    5000,
    10000,
]

ALL_TIME_POINTS = sorted(
    set.union(*[set(v.time_points) for v in BUILT_VERSIONS.values()])
    | {f"{y}q1" for y in range(1975, 2026)}
)


class CRS(BaseModel):
    name: str
    short_name: str
    bounds: tuple[float, float, float, float]
    code: str = ""
    proj_string: str = ""

    @model_validator(mode="after")
    def validate_code_or_proj_string(self) -> "CRS":
        if not self.code and not self.proj_string:
            msg = "Either code or proj_string must be provided."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_code_and_proj_string(self) -> "CRS":
        if self.code and self.proj_string:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code_proj = pyproj.CRS.from_user_input(self.code).to_proj4()
                proj_proj = pyproj.CRS.from_user_input(self.proj_string).to_proj4()
            if code_proj != proj_proj:
                msg = "code and proj_string must represent the same CRS."
                raise ValueError(msg)
        return self

    def to_string(self) -> str:
        if self.code:
            return self.code
        return self.proj_string

    def to_pyproj(self) -> pyproj.CRS:
        if self.code:
            return pyproj.CRS.from_user_input(self.code)
        return pyproj.CRS.from_user_input(self.proj_string)

    def __hash__(self) -> int:
        return hash(self.name)


CRSES: dict[str, CRS] = {
    "wgs84": CRS(
        name="WGS84",
        short_name="wgs84",
        code="EPSG:4326",
        proj_string="+proj=longlat +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "wgs84_anti_meridian": CRS(
        name="WGS84 Anti-Meridian",
        short_name="wgs84_am",
        proj_string="+proj=longlat +lon_0=180 +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "itu_anti_meridian": CRS(
        name="PDC Mercator",
        short_name="itu_am",
        code="EPSG:3832",
        proj_string="+proj=merc +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-5711803.07, -8362698.55, 15807367.69, 10023392.49),
    ),
    "mollweide": CRS(
        name="Mollweide",
        short_name="mollweide",
        code="ESRI:54009",
        proj_string="+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "mollweide_anti_meridian": CRS(
        name="Mollweide Anti-Meridian",
        short_name="mollweide_am",
        proj_string="+proj=moll +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "world_cylindrical": CRS(
        name="World Cylindrical",
        short_name="world_cylindrical",
        code="ESRI:54034",
        proj_string="+proj=cea +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "world_cylindrical_anti_meridian": CRS(
        name="World Cylindrical Anti-Meridian",
        short_name="world_cylindrical_am",
        proj_string="+proj=cea +lat_ts=0 +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "web_mercator": CRS(
        name="Web Mercator",
        short_name="web_mercator",
        code="EPSG:3857",
        proj_string="+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs",
        bounds=(-20037508.34, -20048966.1, 20037508.34, 20048966.1),
    ),
}

# Add some aliases
CRSES["equal_area"] = CRSES["world_cylindrical"]
CRSES["equal_area_anti_meridian"] = CRSES["world_cylindrical_anti_meridian"]
