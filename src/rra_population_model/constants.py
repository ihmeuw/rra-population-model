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


# Open Building Map is a static snapshot, so it ships as a single vintage. The
# provider name is lowercase and hyphen-free because features are addressed as
# `{provider}_{measure}` and split on underscores.
OBM_VERSION = "2025-04-04"
OBM_PROVIDER = "obm_2025q2"
# Real feature files are written here and every other time point links to them.
# 2025q2 is the quarter the 2025-04-04 snapshot actually falls in.
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

# Pixels whose parent densities sum above this are rescaled by 1/total. See
# derive_features for why the tolerance is not simply zero.
OBM_RESCALE_TOLERANCE = 1e-6

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
