import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterra as rt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rra_tools.plotting import strip_axes
from skimage import exposure

from rra_population_model.data import (
    PopulationModelData,
)


def raster_and_gdf_plot(
    fig: Figure, raster_data: rt.RasterArray, gdf: gpd.GeoDataFrame, column: str
) -> None:
    axes = fig.subplots(1, 2)

    ax = axes[0]
    raster_data.plot(ax=ax)
    strip_axes(ax)

    ax = axes[1]
    gdf.plot(ax=ax, column=f"admin_{column}")
    ax.set_xlim(raster_data.x_min, raster_data.x_max)
    ax.set_ylim(raster_data.y_min, raster_data.y_max)

    strip_axes(ax)
    fig.suptitle(column)


def make_tile_diagnostics(tile_key: str) -> None:
    pm_data = PopulationModelData()
    model_frame = pm_data.load_modeling_frame("100")
    tile_meta = model_frame[model_frame.tile_key == tile_key]
    block_key = tile_meta.block_key.iloc[0]
    tile_poly = tile_meta.geometry.iloc[0]
    features = pm_data.load_features(block_key, "2020q1", tile_poly)  # type: ignore[attr-defined]
    if "night_time_lights_alt" in features:
        del features["night_time_lights_alt"]

    gdf = pm_data.load_people_per_structure(tile_key)

    bd_raster = features.pop("building_density")
    features.pop("population")
    pd_raster = features.pop("population_density")

    fig = plt.figure(figsize=(30, 15), layout="constrained")
    sub_figs = fig.subfigures(
        nrows=1,
        ncols=2,
        wspace=0.05,
    )

    n_features = len(features)
    n_rows = min(4, n_features)
    n_cols = n_features // 4 + 1

    feature_figs = sub_figs[1].subfigures(n_rows, n_cols)

    bd_increments = sorted(
        [
            int(k.split("_")[-1].split("m")[0])
            for k in features
            if "building_density" in k
        ]
    )
    feature_order = [f"building_density_average_{inc}m" for inc in bd_increments]
    feature_order += sorted([f for f in features if "building_density" not in f])

    for i, f in enumerate(feature_order):
        row, col = i % 4, i // 4
        sf = feature_figs[row, col]
        raster_and_gdf_plot(sf, features[f], gdf, f)

    main_fig = sub_figs[0].subfigures(2, 1)

    top_fig = main_fig[0].subfigures(1, 2)

    raster_and_gdf_plot(top_fig[0], bd_raster, gdf, "building_density")
    raster_and_gdf_plot(top_fig[1], pd_raster, gdf, "population_density")

    axes = main_fig[1].subplots(1, 2)
    ax = axes[0]
    occ_rate = 100 * pd_raster / bd_raster
    occ_rate._ndarray[~np.isfinite(occ_rate._ndarray)] = 0  # noqa: SLF001
    occ_rate.plot(ax=ax, vmin=0.01, vmax=5, under_color="grey")
    strip_axes(ax)

    ax = axes[1]
    z = gdf[["geometry", "admin_occupancy_rate"]].copy()
    z["admin_occupancy_rate"] *= 100
    z.plot(ax=ax, column="admin_occupancy_rate", vmax=5, vmin=0)
    z[z.admin_occupancy_rate < 0].plot(ax=ax, color="red")
    ax.set_xlim(pd_raster.x_min, pd_raster.x_max)
    ax.set_ylim(pd_raster.y_min, pd_raster.y_max)
    strip_axes(ax)
    main_fig[1].suptitle("Occupancy Rate")


def hist_eq_plot(
    x: np.ndarray,  # type: ignore[type-arg]
    y: np.ndarray,  # type: ignore[type-arg]
    ax: Axes,
) -> None:
    bins = 256
    val, xx, yy = np.histogram2d(x, y, bins=bins)
    val = val.T[::-1] / val.max()
    mask = val == 0
    val_eq = exposure.equalize_hist(1000 * val, nbins=100000, mask=~mask)  # type: ignore[no-untyped-call]
    ax.imshow(
        np.ma.masked_array(val_eq, mask),  # type: ignore[no-untyped-call]
        cmap="cividis",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
    )
