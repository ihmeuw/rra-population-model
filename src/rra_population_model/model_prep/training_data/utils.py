from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
import tqdm
from rasterio import features
from scipy import ndimage
from shapely import area, box, intersection

from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.model_prep.training_data.metadata import (
    TileMetadata,
    TrainingMetadata,
)


def get_intersecting_admins(
    tile_meta: TileMetadata,
    iso3_time_point_list: list[list[str]],
    pm_data: PopulationModelData,
) -> gpd.GeoDataFrame:
    admin_data = []
    for iso3, time_point in iso3_time_point_list:
        year = time_point.split("q")[0]
        a = pm_data.load_census_data(iso3, year, tile_meta.polygon)
        # Need to intersect again with the tile poly because we load based on the
        # intersection with the bounding box.
        is_max_admin = a.admin_level == a.admin_level.max()
        intersects_tile = a.intersects(tile_meta.polygon)
        a = a.loc[is_max_admin & intersects_tile]
        a['census_time_point'] = time_point
        admin_data.append(a)

    admins = pd.concat(admin_data, ignore_index=True)
    admins = admins.rename(
        columns={
            "shape_id": "admin_id",
            "population_total": "admin_population",
        }
    )
    admins["admin_area"] = admins.area
    admins["geometry"] = admins.buffer(0)
    return admins.loc[:, ["admin_id", "admin_population", "admin_area", "geometry", "census_time_point"]]


def get_data_locations_and_years(
    pm_data: PopulationModelData,
) -> list[tuple[str, str, str]]:
    """Get the locations and years for which we have training data."""
    available_census_years = pm_data.list_census_data()
    available_census_years = [
        i for i in available_census_years
        if i[0] in ["MEX", "USA"] and i[1] == "2020"
    ]
    return available_census_years


def build_arg_list(
    resolution: str,
    pm_data: PopulationModelData,
    buffer_size: int | float = 5000,
) -> list[tuple[str, str, str]]:
    modeling_frame = pm_data.load_modeling_frame(resolution)
    data_years = get_data_locations_and_years(pm_data)

    tile_keys_and_times = []
    for iso3, year, quarter in data_years:
        print(f"Processing {iso3} {year}q{quarter}")
        shape = pm_data.load_census_data(iso3, year)
        a1 = (
            shape.loc[shape.admin_level == 1]
            .explode(index_parts=True)
            .convex_hull.buffer(buffer_size)
            .union_all()
        )
        a1_intersection = modeling_frame[modeling_frame.intersects(a1)]
        for tile_key in a1_intersection.tile_key.unique():
            tile_keys_and_times.append(
                pd.DataFrame(
                    {"iso3_time_point": f"{year}q{quarter}", "iso3": iso3},
                    index=pd.Index([tile_key], name='tile_key'),
                )
            )
    tile_keys_and_times = pd.concat(tile_keys_and_times)
    tile_keys_and_times['time_point'] = tile_keys_and_times['iso3_time_point']
    tile_keys_and_times['iso3_time_point'] = (
        tile_keys_and_times['iso3'] + ':' + tile_keys_and_times['iso3_time_point']
    )
    tile_keys_and_times = (
        tile_keys_and_times
        .groupby(['tile_key', 'time_point'])['iso3_time_point']
        .apply(lambda x: ','.join(x.to_list()))
        .sort_index()
    )

    to_run = [
        (tile_key, time_point, iso3_time_points)
        for (tile_key, time_point), iso3_time_points in tile_keys_and_times.items()
    ]
    return to_run


def build_summary_people_per_structure(
    pm_data: PopulationModelData,
    resolution: str,
) -> pd.DataFrame:
    tile_dirs = list(pm_data.tile_training_data_root(resolution).iterdir())
    data = pd.concat(
        [
            pm_data.load_people_per_structure(resolution, tile_dir.name)
            for tile_dir in tqdm.tqdm(tile_dirs)
        ],
        ignore_index=True,
    )
    return data  # type: ignore[no-any-return]


def safe_divide(
    a: npt.NDArray[np.floating[Any]] | pd.DataFrame,
    b: npt.NDArray[np.floating[Any]] | pd.DataFrame,
) -> npt.NDArray[np.floating[Any]]:
    """Divide two arrays, but return 0 where both arrays are 0."""

    if not np.issubdtype(a.dtype, np.floating) or not np.issubdtype(  # type: ignore[arg-type]
        b.dtype,  # type: ignore[arg-type]
        np.floating,
    ):
        msg = "Both arrays must be floating point."
        raise TypeError(msg)

    mask = ~((a == 0) & (b == 0))
    r = np.zeros_like(a)
    r[mask] = a[mask] / b[mask]
    return r


def _get_valid_data_mask(
    array: npt.NDArray[np.floating[Any]], no_data_value: Any
) -> npt.NDArray[np.bool_]:
    """Pixels carrying real data, given a raster's nodata convention."""
    if no_data_value is None:
        valid = np.ones(array.shape, dtype=bool)
    elif isinstance(no_data_value, float) and np.isnan(no_data_value):
        valid = ~np.isnan(array)
    else:
        valid = array != no_data_value
    return np.asarray(valid, dtype=bool)


def build_overlay_skeleton(
    shapes: gpd.GeoDataFrame,
    template_raster: rt.RasterArray,
    *,
    drop_no_data: bool = True,
    dilate_border: bool = True,
) -> pd.DataFrame:
    """Map polygons onto raster pixels by rasterization, not polygon overlay.

    Returns one row per (shape, pixel) intersection with ``pixel_id``,
    ``shape_id`` and ``isection_area``. ``shape_id`` is the 0-based row position
    into ``shapes``, not any string identifier the caller may hold - integer
    codes keep the downstream groupby off string hashing.

    ``pixel_id`` is the C-order flat index into the template raster's array,
    which is the same index ``RasterArray.to_gdf().reset_index()`` produces, so
    per-pixel columns can be attached by position without a spatial join.

    Why this rather than ``GeoDataFrame.overlay``: overlay's cost scales with
    the *area* of the two frames, because every pixel has to become a polygon
    first. Here only pixels a shape boundary passes through need geometry at
    all - interior pixels lie wholly within one shape, so a rasterized centre
    hit gives exact full coverage for free. Cost scales with perimeter instead.

    Adapted from ``postprocess.census_rake.utils.build_overlay_skeleton``. The
    two are kept separate until this path is validated against the overlay it
    replaces; consolidating them is a follow-up.

    Parameters
    ----------
    shapes
        Polygons to intersect, in the raster's CRS.
    template_raster
        Supplies the grid: shape, transform, resolution and nodata.
    drop_no_data
        Drop pixels the raster marks as nodata. Census raking does this so that
        in-shape water yields NaN rather than a spurious zero. ``overlay`` on a
        ``to_gdf`` frame keeps them, so this defaults on but is switchable while
        the two paths are being compared.
    dilate_border
        Widen the border mask by one cell before splitting interior from border.

        ``all_touched`` is a Bresenham-style traversal, not an exact "does this
        line clip this cell" test, so a boundary running nearly tangent to a
        pixel edge can leave a cell it genuinely crosses unmarked. That cell then
        takes the interior shortcut and is awarded the *whole* pixel, while the
        shape on the far side loses its sliver entirely - area is misattributed
        between admins rather than lost, so tile totals barely move.

        On by default. Measured on a stratified sample of the training
        population (40 tiles across four admin-count strata, plus 38 more in
        the low-admin strata):

        =================  ==============  ==============  ==============
        admins per tile    geometry off    occupancy off   worst sliver
        =================  ==============  ==============  ==============
        1-10               2/21            0/21            6.66 m2
        11-100             3/37            0/37            3.94 m2
        101-1,000          4/10            1/10            5.74 m2
        >1,000             6/10            6/10            9.40 m2
        =================  ==============  ==============  ==============

        Weighted to the population, roughly 25% of tiles deviate geometrically
        without it and 9% move their occupancy rate; with it, 0 of the 78
        sampled tiles deviate at all. Worst deviation in the rates the model
        trains on was 4.8e-03 (pixel) and 5.8e-04 (admin). The error is
        systematic - always toward the admin holding the pixel centre - so it
        does not average out.

        Cost, anchored on jobmon workflow 523886 (the overlay-method run that
        produced ``training-data-OLD-v8``: 34,230 tasks, 3h07m, median task
        60s). Median task time here is 27.1s without the dilation and 29.5s
        with it, so the whole run lands near 1h25m and 1h33m respectively -
        still about twice as fast as the overlay it replaces. Total compute
        goes from 386 to 492 task-hours.

        Set it False to trade that back: the cost falls disproportionately on
        tiles with few admins, which have large tile neighbourhoods and never
        moved occupancy in 58 samples. A threshold - dilating only above ~100
        admins - captured all of the observed benefit for about 15% of the
        cost, if runtime ever becomes binding.

        Peak memory is unchanged: 1.00x on four stress tiles, chosen for the
        conditions that would show it worst - neighbourhoods up to 88 tiles and
        up to 39,053 admins. The border-pixel working set does grow about 1.7x,
        but it is not what sets the peak, so no task crosses a
        ``resource_scales`` boundary. Worth re-checking if the peak's driver
        changes: the overlay run reached 163 GiB against a 10 GiB median, so
        the headroom above the median is where a regression would surface.
    """
    out_shape = template_raster.shape
    transform = template_raster.transform
    pixel_area = np.float32(
        abs(template_raster.x_resolution * template_raster.y_resolution)
    )

    if drop_no_data:
        valid = _get_valid_data_mask(
            template_raster.to_numpy().astype(np.float32),
            template_raster.no_data_value,
        )
    else:
        valid = np.ones(out_shape, dtype=bool)

    shapes = shapes.reset_index(drop=True)
    n_shapes = len(shapes)
    id_dtype = "uint16" if n_shapes < np.iinfo(np.uint16).max else "uint32"

    # Burn shape ids (1-based; 0 == no shape) by pixel centre, then drop nodata.
    assigned = features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(shapes.geometry.to_numpy())),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=id_dtype,
    )
    assigned[~valid] = 0

    # Any pixel a shape boundary touches needs an exact intersection.
    border_mask = features.rasterize(
        ((geom, 1) for geom in shapes.geometry.boundary.to_numpy()),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    if dilate_border:
        # A grazed cell neighbours one the line does cross, so widening the mask
        # by a cell recovers it. That is the rationale rather than a proof; what
        # is checked is the outcome, against an exhaustive float64 overlay,
        # which this then matches to float32 storage precision. 8-connectivity
        # because a proof that 4 suffices is not available and it costs ~7% more.
        # See the docstring for what this is worth and what it costs.
        border_mask = ndimage.binary_dilation(
            border_mask, structure=np.ones((3, 3), dtype=bool)
        )
    border_mask &= valid

    # Interior: wholly inside one shape, so full coverage and no geometry work.
    interior_idx = np.flatnonzero((assigned > 0) & ~border_mask)
    interior = pd.DataFrame(
        {
            "pixel_id": interior_idx,
            "shape_id": (assigned.ravel()[interior_idx] - 1).astype(id_dtype),
            "isection_area": pixel_area,
        }
    )

    # Border: build cell polygons here only, and intersect them with the shapes.
    border_idx = np.flatnonzero(border_mask)
    if border_idx.size:
        rows, cols = np.unravel_index(border_idx, out_shape)
        a, b, c, d, e, f = (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        )
        x0 = a * cols + b * rows + c
        y0 = d * cols + e * rows + f
        x1 = a * (cols + 1) + b * (rows + 1) + c
        y1 = d * (cols + 1) + e * (rows + 1) + f
        boxes = box(
            np.minimum(x0, x1),
            np.minimum(y0, y1),
            np.maximum(x0, x1),
            np.maximum(y0, y1),
        )
        border_pixels = gpd.GeoDataFrame(
            {"pixel_id": border_idx}, geometry=boxes, crs=template_raster.crs
        )
        # Pair border pixels with the shapes they hit, then intersect vectorized
        # and take the area. This sidesteps overlay's noding/polygonize
        # machinery, which is far slower on convoluted boundaries; zero-area
        # (line or point) touches fall out through the area > 0 filter.
        joined = gpd.sjoin(
            border_pixels,
            shapes[["geometry"]],
            predicate="intersects",
            how="inner",
        )
        shape_geom = shapes.geometry.to_numpy()[joined["index_right"].to_numpy()]
        isection_area = area(intersection(joined.geometry.to_numpy(), shape_geom))
        border = pd.DataFrame(
            {
                "pixel_id": joined["pixel_id"].to_numpy(),
                "shape_id": joined["index_right"].to_numpy().astype(id_dtype),
                "isection_area": isection_area.astype(np.float32),
            }
        )
        border = border[border["isection_area"] > 0]
    else:
        border = pd.DataFrame(
            {
                "pixel_id": np.array([], dtype=np.int64),
                "shape_id": np.array([], dtype=id_dtype),
                "isection_area": np.array([], dtype=np.float32),
            }
        )

    overlay = pd.concat([interior, border], ignore_index=True)
    overlay["pixel_area"] = pixel_area
    return overlay


def get_tile_feature_gdf(
    tile_meta: TileMetadata,
    training_meta: TrainingMetadata,
    pm_data: PopulationModelData,
    time_point: str,
) -> pd.DataFrame:
    """Intersect the tile's admins with its pixels, and attach the features.

    Returns one row per (admin, pixel) intersection. No geometry: the pixel
    polygons this used to build were never read downstream - `process_model_gdf`
    works purely off `isection_area` / `pixel_area` / `admin_area`, and
    `filter_to_admin_gdf` re-attaches admin geometry by `admin_id`.

    The intersection is done by rasterizing the admins onto the pixel grid
    rather than by `GeoDataFrame.overlay`. Overlay costs scale with the *area*
    of both frames, since every pixel first has to become a polygon; rasterizing
    means only pixels an admin boundary passes through need geometry at all. See
    `build_overlay_skeleton`.
    """
    tile_features = {}
    for feature_name in training_meta.features:
        tile_features[feature_name] = pm_data.load_feature(
            resolution=training_meta.resolution,
            block_key=tile_meta.block_key,
            feature_name=feature_name,
            time_point=time_point,
            subset_bounds=tile_meta.polygon,
        )

    default_raster = training_meta.denominators[0]
    bd_raster = tile_features.pop(default_raster)

    # Per-pixel features, keyed by the flat C-order index into the tile raster -
    # the same index `build_overlay_skeleton` emits, so these attach by position
    # rather than by a spatial join.
    pixel_df = pd.DataFrame(
        {
            "pixel_id": np.arange(bd_raster.size),
            f"pixel_{default_raster}": bd_raster.to_numpy().flatten(),
        }
    )
    for feature_name, feature_raster in tile_features.items():
        pixel_df[f"pixel_{feature_name}"] = feature_raster.to_numpy().flatten()

    # `intersecting_admins` is already restricted to admins meeting this tile,
    # and the raster covers exactly the tile, so pixels outside it do not exist.
    # The clip to a buffered tile the overlay path needed is therefore dropped.
    admins = training_meta.intersecting_admins.reset_index(drop=True)

    # `drop_no_data=False` keeps nodata pixels, matching what `to_gdf` did.
    # Dropping them would leave `raster_from_pixel_feature` to fill the gaps with
    # 0.0, which differs from today for the mean-aggregated features.
    skeleton = build_overlay_skeleton(admins, bd_raster, drop_no_data=False)

    admin_cols = admins.drop(columns="geometry")
    tile_gdf = (
        skeleton.join(admin_cols, on="shape_id")
        .merge(pixel_df, on="pixel_id", how="left")
        .drop(columns="shape_id")
    )
    tile_gdf["block_key"] = tile_meta.block_key
    tile_gdf["tile_key"] = tile_meta.key
    tile_gdf["time_point"] = time_point
    return tile_gdf


def process_model_gdf(
    model_gdf: gpd.GeoDataFrame,
    training_meta: TrainingMetadata,
) -> gpd.GeoDataFrame:
    """Calculate admin and pixel level properties from an intersection GeoDataFrame.

    Parameters
    ----------
    model_gdf
        The GeoDataFrame of admin/pixel/intersection properties.
    features
        The list of features beyond population, occupancy rate, and building
        density to process.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame with all admin and pixel level properties.
    """
    min_density = 0.01
    pixel_area = model_gdf["pixel_area"].iloc[0]
    min_admin_density = min_density * pixel_area

    model_gdf["admin_population_density"] = (
        model_gdf["admin_population"] / model_gdf["admin_area"]
    )
    model_gdf["admin_area_weight"] = (
        model_gdf["isection_area"] / model_gdf["admin_area"]
    )
    model_gdf["pixel_area_weight"] = (
        model_gdf["isection_area"] / model_gdf["pixel_area"]
    )

    model_gdf["admin_multi_tile"] = (
        model_gdf.groupby("admin_id")["tile_key"].transform("nunique") > 1
    ).astype(float)
    model_gdf["pixel_multi_tile"] = model_gdf["admin_multi_tile"]

    for denominator in training_meta.denominators:
        denominator_df = model_gdf[
            ["tile_key", "pixel_id", "admin_id", "admin_population"]
        ].copy()
        denominator_df["isection_built"] = (
            model_gdf[f"pixel_{denominator}"] * model_gdf["isection_area"]
        )
        denominator_df["pixel_built"] = (
            model_gdf[f"pixel_{denominator}"] * model_gdf["pixel_area"]
        )
        denominator_df["admin_built"] = denominator_df.groupby("admin_id")[
            "isection_built"
        ].transform("sum")
        # if denominator.startswith("microsoft"):
        #     low_density = denominator_df["admin_built"] < min_admin_density
        #     denominator_df.loc[low_density, "admin_built"] = 0.0
        #     denominator_df.loc[low_density, "isection_built"] = 0.0
        # elif not denominator.startswith("ghsl"):
        #     raise ValueError(f"Unexpected denominator: {denominator}")

        denominator_df[f"admin_{denominator}"] = safe_divide(
            denominator_df["admin_built"], model_gdf["admin_area"]
        )

        denominator_df["admin_built_weight"] = safe_divide(
            denominator_df["isection_built"], denominator_df["admin_built"]
        )
        denominator_df["pixel_built_weight"] = safe_divide(
            denominator_df["isection_built"], denominator_df["pixel_built"]
        )

        denominator_df["isection_population"] = (
            denominator_df["admin_population"] * denominator_df["admin_built_weight"]
        )
        denominator_df["pixel_population"] = denominator_df.groupby(
            ["tile_key", "pixel_id"]
        )["isection_population"].transform("sum")

        # ADMIN OCCUPANCY RATE
        pos_mask = (
            (denominator_df["admin_population"] > 0)
            & (denominator_df["admin_built"] > 0)
        )

        admin_occupancy_rate = safe_divide(
            denominator_df["admin_population"].astype(float),
            denominator_df["admin_built"],
        )
        denominator_df["admin_occupancy_rate"] = np.nan
        denominator_df.loc[pos_mask, "admin_occupancy_rate"] = admin_occupancy_rate[pos_mask]
        denominator_df["admin_log_occupancy_rate"] = np.nan
        denominator_df.loc[pos_mask, "admin_log_occupancy_rate"] = np.log(
            admin_occupancy_rate[pos_mask]
        )

        # PIXEL OCCUPANCY RATE
        pos_mask = (
            (denominator_df["pixel_population"] > 0)
            & (denominator_df["pixel_built"] > 0)
        )

        pixel_occupancy_rate = safe_divide(
            denominator_df["pixel_population"],
            denominator_df["pixel_built"],
        )
        denominator_df["pixel_occupancy_rate"] = np.nan
        denominator_df.loc[pos_mask, "pixel_occupancy_rate"] = pixel_occupancy_rate[pos_mask]
        denominator_df["pixel_log_occupancy_rate"] = np.nan
        denominator_df.loc[pos_mask, "pixel_log_occupancy_rate"] = np.log(
            pixel_occupancy_rate[pos_mask]
        )

        keep_measures = [
            "admin_built",
            "admin_occupancy_rate",
            "admin_log_occupancy_rate",
            "pixel_built",
            "pixel_built_weight",
            "pixel_population",
            "pixel_occupancy_rate",
            "pixel_log_occupancy_rate",
        ]

        for measure in keep_measures:
            model_gdf[f"{measure}_{denominator}"] = denominator_df[measure]

    for feature in training_meta.features:
        model_gdf[f"admin_{feature}"] = (
            model_gdf[f"pixel_{feature}"] * model_gdf["admin_area_weight"]
        )
        model_gdf[f"admin_{feature}"] = model_gdf.groupby("admin_id")[
            f"admin_{feature}"
        ].transform("sum")

    return model_gdf


def filter_to_admin_gdf(
    model_gdf: gpd.GeoDataFrame,
    training_meta: TrainingMetadata,
) -> gpd.GeoDataFrame:
    """Filter the model GDF to only the admin-level features and rows."""
    keep_cols = ["block_key", "tile_key", "time_point"] + [
        c for c in model_gdf if c[:5] == "admin"
    ]
    admin_df = model_gdf.loc[:, keep_cols].groupby("admin_id").first().reset_index()
    # Geometry comes from the admin shapes, not from the intersection rows - it
    # always did; the old code carried a geometry column through only to
    # overwrite it here.
    geometry = (
        training_meta.intersecting_admins.set_index("admin_id")
        .loc[admin_df["admin_id"], "geometry"]
        .to_numpy()
    )
    return gpd.GeoDataFrame(
        admin_df, geometry=geometry, crs=training_meta.intersecting_admins.crs
    )


def raster_from_pixel_feature(
    tile_gdf: gpd.GeoDataFrame,
    feature_name: str,
    raster_template: rt.RasterArray,
) -> rt.RasterArray:
    """Create a raster from a pixel feature in the tile GeoDataFrame.

    Parameters
    ----------
    tile_gdf
        The GeoDataFrame of the pixel features.
    feature_name
        The name of the feature to convert to a raster.
    raster_template
        The template raster to use for the output raster.

    Returns
    -------
    rt.RasterArray
        The raster of the pixel feature.
    """
    if feature_name.startswith("population") or feature_name == "area_weight":
        agg_func = "sum"
    elif (
        feature_name.startswith("occupancy_rate")
        or feature_name.startswith("log_occupancy_rate")
    ):
        agg_func = "mean"
    elif feature_name == "multi_tile":
        agg_func = "first"
    else:
        value_error = f"Unexpected feature name: {feature_name}"
        raise ValueError(value_error)

    idx = np.arange(raster_template.size)
    feature_data = (
        tile_gdf.groupby("pixel_id")[f"pixel_{feature_name}"]
        .agg(agg_func)
        .reindex(idx, fill_value=0.0)
        .to_numpy()
        .astype(np.float32)
        .reshape(raster_template.shape)
    )
    feature_raster = rt.RasterArray(
        data=feature_data,
        transform=raster_template.transform,
        crs=raster_template.crs,
        no_data_value=np.nan,
    )
    return feature_raster
