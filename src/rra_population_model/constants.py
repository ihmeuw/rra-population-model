import itertools
import warnings
from pathlib import Path

import pyproj
from pydantic import BaseModel, model_validator

RRA_ROOT = Path("/mnt/team/rapidresponse/")
RRA_CREDENTIALS_ROOT = RRA_ROOT / "priv" / "shared" / "credentials"
RRA_BINARIES_ROOT = RRA_ROOT / "priv" / "shared" / "bin"
GEOSPATIAL_COVARIATES_ROOT = Path("/snfs1/WORK/11_geospatial/01_covariates")
BUILDING_DENSITY_ROOT = RRA_ROOT / "pub" / "building-density"
POPULATION_DATA_ROOT = RRA_ROOT / "pub" / "population" / "data" / "02-processed-data"
MODEL_ROOT = RRA_ROOT / "pub" / "population-model"

RESOLUTIONS = ["40", "100", "250", "500", "1000"]

FEATURE_AVERAGE_RADII = [
    100,
    500,
    1000,
    2500,
    5000,
    10000,
]

GHSL_TIME_POINTS = [f"{y}q1" for y in range(1975, 2024)]
MICROSOFT_TIME_POINTS = {
    "microsoft_v2": [
        f"{y}q{q}" for q, y in itertools.product(range(1, 5), range(2018, 2024))
    ][:-1],
    "microsoft_v3": ["2023q3"],
    "microsoft_v4": ["2023q4"],
}
ALL_TIME_POINTS = sorted(
    set(GHSL_TIME_POINTS) | set().union(*MICROSOFT_TIME_POINTS.values())
)

ANNUAL_TIME_POINTS = [f"{y}" for y in range(1950, 2101)]


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
