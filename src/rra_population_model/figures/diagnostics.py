"""Frozen diagnostics figures.

This module contains functions that load all their own data and plot a single
diagnostic.  Some of these are one-off descriptive plots that may be useful later,
others may be folded into a standard diagnostic tool once we get those set up.
"""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rra_population_model.data import (
    PopulationModelData,
    RRAPopulationData,
)


def plot_area_distribution() -> None:
    """Plot a histogram of the area distribution of the population
    Mexico has a bunch of very small admin units.  This plot shows the distribution
    of the smallest 50% of admin unit areas with a comparison to the modeling
    pixel size.
    """
    pm_data = PopulationModelData()
    gdf = gpd.read_parquet(pm_data.training_data / "all.parquet")  # type: ignore[attr-defined]
    area = gdf.admin_area
    lower = area < area.quantile(0.50)
    fig, ax = plt.subplots(figsize=(5, 3))
    area[lower].plot.hist(bins=100, ax=ax)
    ax.axvline(x=40 * 40, color="k")
    ax.text(x=40 * 40 * 1.1, y=5000, s="Pixel Area", rotation="vertical")
    ax.set_xlabel("Admin unit area ($m^2$)")
    ax.set_title("Smallest 50% of admin units in MEX")


def plot_building_density_histogram(tile_key: str) -> None:
    pop_data = RRAPopulationData()
    pm_data = PopulationModelData()
    modeling_frame = pm_data.load_modeling_frame("100")
    tile_poly = modeling_frame[modeling_frame.tile_key == tile_key].geometry.iloc[0]
    bdi = pop_data.load_building_density_index().to_crs(modeling_frame.crs)  # type: ignore[attr-defined]
    bd_tiles = bdi[bdi.intersects(tile_poly)].quad_name.tolist()
    bd = pop_data.load_building_density_tiles(  # type: ignore[attr-defined]
        "2020q1",
        bd_tiles,
    ).set_no_data_value(np.nan)

    bds = pd.Series(bd.to_numpy().flatten())
    binwidth = 0.00025
    cuts = [0, 0.0025, 0.005, 0.1]
    fig, axes = plt.subplots(figsize=(12, 4), ncols=len(cuts) - 1)

    for col, (cmin, cmax) in enumerate(zip(cuts[:-1], cuts[1:], strict=False)):
        bins = int((cmax - cmin) / binwidth)
        axes[col].hist(bds[(cmin < bds) & (bds <= cmax)], bins=bins)
        axes[col].set_title(f"{cmin} - {cmax}")

    fig.tight_layout()


def compare_kenya_and_iceland_resolution() -> None:
    """Shows the resolution discrepancy by latitude induced by the projection choice."""
    pop_data = RRAPopulationData()
    t1 = pop_data.load_building_density_tile(  # type: ignore[attr-defined]
        quarter="2020q1", quad_name="L15-0899E-1503N"
    )
    threshold = 0.01
    t1._ndarray[t1._ndarray < threshold] = np.nan  # noqa: SLF001
    t2 = pop_data.load_building_density_tile(  # type: ignore[attr-defined]
        quarter="2020q1", quad_name="L15-1233E-1016N"
    )
    t2._ndarray[t2._ndarray < threshold] = np.nan  # noqa: SLF001

    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
    ax = axes[0]
    t1_subset = t1[150:180, 260:290]
    t1_subset_gdf = t1_subset.to_gdf()
    t1_subset_gdf.boundary.plot(ax=ax, color="k", linewidth=0.5)

    cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery)

    ax = axes[1]
    t2_subset = t2[208:222, 245:260]
    t2_subset_gdf = t2_subset.to_gdf()
    t2_subset_gdf.boundary.plot(ax=ax, color="k")

    cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery)


def examine_impact_of_throwing_out_multi_tile_admins() -> None:
    pm_data = PopulationModelData()
    admin_data = pm_data.load_people_per_structure(resolution="100")
    admin_data["admin_is_multi_tile"] = (
        admin_data.groupby("admin_id")["tile_key"].transform("nunique") > 1
    )

    first = ~admin_data["admin_id"].duplicated()
    multi_tile = admin_data["admin_is_multi_tile"].astype(bool)
    has_people = admin_data["admin_population"] > 0
    has_buildings = admin_data["admin_built_square_meters"] > 16  # noqa: PLR2004
    fmt_string = "{0:<20} {1:>15} {2:<10.2f} {3:>15} {4:<10.2f} {5:<10}"

    mask_map = {
        "Total": first,
        "Multi-tile": first & multi_tile,
        "Populated": first & has_people,
        "Built": first & has_buildings,
        "MT populated": first & multi_tile & has_people,
        "MT built": first & multi_tile & has_buildings,
    }

    total_units = len(admin_data[first])
    total_pop = admin_data.loc[first, "admin_population"].sum()

    for label, mask in mask_map.items():
        num_units = mask.sum()
        pop = admin_data.loc[mask, "admin_population"].sum()
        print(
            fmt_string.format(
                label,
                num_units,
                num_units / total_units * 100,
                pop,
                pop / total_pop * 100,
            )
        )

    small = admin_data["admin_population"] < 500  # noqa: PLR2004

    fig, ax = plt.subplots(figsize=(8, 4))

    admin_data.loc[
        first & ~multi_tile & has_people & small, "admin_population"
    ].sort_values().hist(bins=50, alpha=0.5, color="firebrick", ax=ax, density=True)
    admin_data.loc[
        first & multi_tile & has_people & small, "admin_population"
    ].sort_values().hist(bins=50, alpha=0.5, color="dodgerblue", ax=ax, density=True)
    ax.set_xlabel("Admin Population")
    ax.set_ylabel("Proportion")

    plt.show()

    small = admin_data["admin_area"] < 1e9  # noqa: PLR2004

    fig, ax = plt.subplots(figsize=(8, 4))

    admin_data.loc[first & small, "admin_area"].sort_values().hist(
        bins=50, alpha=0.5, color="firebrick", ax=ax, density=True
    )
    admin_data.loc[first & multi_tile & small, "admin_area"].sort_values().hist(
        bins=50, alpha=0.5, color="dodgerblue", ax=ax, density=True
    )
    ax.set_xlabel("Admin Area")
    ax.set_ylabel("Proportion")
    plt.show()

    print(
        pd.concat(
            [
                admin_data.loc[first & has_people, "admin_area"]
                .describe(percentiles=[0.9, 0.95, 0.99, 0.995, 0.9995])
                .rename("populated"),
                admin_data.loc[first & ~has_people, "admin_area"]
                .describe(percentiles=[0.9, 0.95, 0.99, 0.995, 0.9995])
                .rename("unpopulated"),
            ],
            axis=1,
        )
    )
