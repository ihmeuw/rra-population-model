"""Code for preparing nighttime lights data for the population model."""

from pathlib import Path

import numpy as np
import rasterra as rt
import shapely
from rasterio.fill import fillnodata

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model_prep.features.metadata import FeatureMetadata


def process_ntl(
    feature_metadata: FeatureMetadata,
    pm_data: PopulationModelData,
) -> None:
    ntl = load_and_format_ntl(feature_metadata)
    pm_data.save_feature(
        ntl,
        feature_name="nighttime_lights",
        **feature_metadata.shared_kwargs,
    )
    log_ntl = np.log(1 + ntl)
    pm_data.save_feature(
        log_ntl,
        feature_name="log_nighttime_lights",
        **feature_metadata.shared_kwargs,
    )


def load_and_format_ntl(
    feature_metadata: FeatureMetadata,
) -> rt.RasterArray:
    year = int(feature_metadata.time_point.split("q")[0])
    year = min(year, 2022)
    year = max(year, 2000)
    if year < 2021:  # noqa: PLR2004
        rel_path = f"57_dmsp_viirs_harm_2/dataverse_files/LongNTL_{year}.tif"
    elif year < 2023:  # noqa: PLR2004
        rel_path = f"57_dmsp_viirs_harm_2/v4_update/LongNTL_{year}.tif"
    else:
        msg = f"No NTL for year {year}"
        raise NotImplementedError(msg)
    path = pmc.GEOSPATIAL_COVARIATES_ROOT / rel_path
    return _load_and_process_ntl(path, feature_metadata)


def _load_and_process_ntl(
    ntl_path: Path,
    feature_metadata: FeatureMetadata,
) -> rt.RasterArray:
    block_gdf = feature_metadata.block_frame
    threshold = 100  # Degrees, just need something bigger than the block

    wgs84 = pmc.CRSES["wgs84"].to_pyproj()
    clip_gdf = block_gdf.dissolve("block_key").to_crs(wgs84)
    xmin, ymin, xmax, ymax = clip_gdf.total_bounds

    if (
        not clip_gdf.is_valid.all()
        or np.isinf(clip_gdf.total_bounds).any()
        or xmax - xmin > threshold
    ):
        bounds = block_gdf.bounds
        xmin, ymin, xmax, ymax = (
            bounds["minx"],
            bounds["miny"],
            bounds["maxx"],
            bounds["maxy"],
        )

        lat_long_bounds = block_gdf.to_crs(wgs84).bounds

        xs_min, xs_max = lat_long_bounds[xmin == xmin.min()].iloc[0][["minx", "maxx"]]
        xe_min, xe_max = lat_long_bounds[xmax == xmax.max()].iloc[0][["minx", "maxx"]]

        if xs_max - xs_min > threshold:  # Left edge ran over
            xs, xe = -180.0, xe_max
        elif xe_max - xe_min > threshold:  # Right edge ran over
            xs, xe = xs_min, 180.0
        else:
            # Sometimes projecting the tiles individually just fixes things
            if (xe_max - xs_min) > threshold:
                msg = "Unknown projection issue"
                raise RuntimeError(msg)
            xs, xe = xs_min, xe_max

        ys = lat_long_bounds[ymin == ymin.min()].miny.iloc[0]
        ye = lat_long_bounds[ymax == ymax.max()].maxy.iloc[0]

        lat_bound, lon_bound = 90, 180
        bad_lon = not (-lon_bound <= xs < xe <= lon_bound) or xe - xs >= threshold
        bad_lat = not (-lat_bound <= ys < ye <= lat_bound) or ye - ys >= threshold
        if bad_lon or bad_lat:
            msg = "Unknown projection issue"
            raise RuntimeError(msg)

        clip_poly = shapely.box(xs, ys, xe, ye)
    else:
        clip_poly = clip_gdf.geometry.iloc[0]

    ntl = (
        rt.load_raster(ntl_path, clip_poly)
        .set_no_data_value(np.nan)
        .resample_to(feature_metadata.block_template, "average")
        .astype(np.float32)
    )
    # The high resolution NTL data in particular has some no data values that need
    # to be filled in, though I have not systematically checked this for all NTL data.
    filled_ntl_data = fillnodata(
        ntl.to_numpy(),
        mask=~ntl.no_data_mask,
    )
    filled_ntl = rt.RasterArray(
        np.nan_to_num(filled_ntl_data),
        transform=ntl.transform,
        crs=ntl.crs,
        no_data_value=ntl.no_data_value,
    )

    return filled_ntl
