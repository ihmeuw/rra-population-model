import types
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPDF2 import PdfMerger
from rra_tools import jobmon
from rra_tools.shell_tools import touch

from rra_population_model.data import (
    PEOPLE_PER_STRUCTURE_ROOT,
    PopulationModelData,
)
from rra_population_pipelines.shared.cli_tools import options as clio
from rra_population_pipelines.shared.constants import CRS
from rra_population_pipelines.shared.plot_utils import strip_axes, write_or_show

DEFAULT_LINEWIDTH = 1.0
CMAP = plt.get_cmap("viridis")


#####################
# Utility functions #
#####################


def round_to_sigfigs(x: float, sigfigs: int) -> float:
    return round(x, sigfigs - int(np.floor(np.log10(abs(x)))) - 1)


######################
# Map plots (page 1) #
######################

# Functions to rework
#####################


def generate_mainland_mask(gdf: gpd.GeoDataFrame) -> list[str]:
    msg = "This function is not yet implemented."
    raise NotImplementedError(msg)
    gdf = gdf.to_crs(CRS.EQUAL_AREA)
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf["geometry"].buffer(1000)

    overlapping_shapes = []

    for _, shape in gdf_buffered.iterrows():
        overlaps = gdf_buffered[gdf_buffered.overlaps(shape.geometry)]

        if not overlaps.empty:
            total_population = overlaps["value"].sum()
            overlapping_shapes.append(
                (shape.geometry, total_population, overlaps["value"].to_numpy())
            )

    if overlapping_shapes:
        overlapping_geometries, populations, original_values = zip(
            *overlapping_shapes, strict=False
        )
        overlapping_gdf = gpd.GeoDataFrame(geometry=list(overlapping_geometries))
    else:
        overlapping_gdf = gpd.GeoDataFrame(geometry=[])

    contained_shapes = []
    if not overlapping_gdf.empty:
        for _, shape in gdf.iterrows():
            if overlapping_gdf.contains(shape.geometry).any():
                contained_shapes.append(shape)

    gdf_filtered = gpd.GeoDataFrame(contained_shapes, columns=gdf.columns)

    shape_ids = gdf_filtered["shape_id"].unique() if not gdf_filtered.empty else []

    return list(map(str, shape_ids))


def filter_by_list(gdf: gpd.GeoDataFrame, shape_ids: list[str]) -> gpd.GeoDataFrame:
    gdf["path_to_top_parent_list"] = gdf["path_to_top_parent"].apply(
        lambda x: x.split(",")
    )
    gdf = gdf[
        gdf["path_to_top_parent_list"].apply(
            lambda x: any(shape_id in x for shape_id in shape_ids)
        )
    ]

    return gdf


# Helpers
#########


def get_census_map_norm(
    data: pd.Series,  # type: ignore[type-arg]
    bins: list[float] | None = None,
) -> mpl.colors.Normalize:
    if bins is None:
        vmin = 0
        vmax = round_to_sigfigs(data.quantile(0.90), 2)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=256)
    return norm


def plot_census_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    max_admin_level: int,
    norm: mpl.colors.Normalize,
    ax: Axes,
) -> Axes:
    most_detailed_mask = gdf["admin_level"] == max_admin_level

    gdf.loc[most_detailed_mask].plot(
        column=column,
        ax=ax,
        legend=False,
        cmap=CMAP,
        norm=norm,
        edgecolor="none",
        rasterized=True,
    )

    zero_value_area = gdf.loc[most_detailed_mask & (gdf[column] == 0)]
    if not zero_value_area.empty and not zero_value_area.geometry.is_empty.all():
        zero_value_area.plot(
            ax=ax,
            color="lightgrey",
            rasterized=True,
        )

    for admin_level in range(max_admin_level - 1, 0, -1):
        mask = gdf["admin_level"] == admin_level
        gdf[mask].boundary.plot(
            ax=ax,
            color="black",
            linewidth=DEFAULT_LINEWIDTH / (2**admin_level),
            rasterized=True,
        )

    strip_axes(ax)

    return ax


def add_census_map_colorbar(title: str, norm: mpl.colors.Normalize, ax: Axes) -> Axes:
    if hasattr(norm, "boundaries"):
        # We want centered ticks with labels that correspond to the bin edges.
        bins = norm.boundaries
        ticks = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        bin_labels = [int(b) for b in bins]
        tick_labels = (
            [f"<{bin_labels[1]:d}"]
            + [
                f"{bin_labels[i]:d}-{bin_labels[i + 1]:d}"
                for i in range(1, len(bin_labels) - 2)
            ]
            + [f">{bin_labels[-2]:d}"]
        )
    else:
        ticks = list(np.linspace(norm.vmin, norm.vmax, num=5))  # type: ignore[arg-type]
        ticks = [round(tick) for tick in ticks]
        tick_labels = [f"{tick:,}" for tick in ticks]
        tick_labels[-1] += "+"

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    cbar = ax.figure.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels, fontsize=12)
    cbar.set_label(title, fontsize=16)

    return ax


# Main function
###############


def plot_admin_level_data(
    census_data: gpd.GeoDataFrame,
    iso3: str,
    year: str,
    save_path: str | Path | None = None,
) -> None:
    # Some countries have territories that are very distant from the mainland.
    # This causes the plot to be very large and mostly empty. To avoid this, we
    # filter out islands and other territories that are far from the mainland.
    split_mainland = ["USA", "FRA"]
    if iso3 in split_mainland:
        on_mainland = generate_mainland_mask(census_data)
        census_data = census_data.loc[on_mainland]
        note_flag = True
    else:
        note_flag = False

    max_admin_level = min(census_data["admin_level"].max(), 3)
    plot_data = census_data.loc[census_data["admin_level"] == max_admin_level]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), layout="constrained")

    plot_specs = [
        ("population_total", "Population Count", None),
        ("population_density", "Population Density", [0, 1, 10, 100, 1000, 10000, 1e6]),
    ]

    for ax, (column, title, bins) in zip(axes, plot_specs, strict=False):
        norm = get_census_map_norm(plot_data[column], bins)
        plot_census_map(census_data, column, max_admin_level, norm, ax)
        add_census_map_colorbar(title, norm, ax)

    main_title = f"{iso3} Census {year} - Admin Level {max_admin_level}" + (
        "*" if note_flag else ""
    )
    fig.suptitle(main_title, fontsize=20)

    if note_flag:
        fig.text(
            0.5,
            0.01,
            "*Some administrative units not shown",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    write_or_show(fig, save_path, format="pdf", dpi=300)


##########################
# Heatmap plots (page 2) #
##########################


def generate_heatmap_data(
    admin_data: pd.DataFrame,
) -> pd.DataFrame:
    population_bins = [-np.inf, 0, 1, 10, 25, 50, 100, 250, 500, 1000, 5000, np.inf]
    area_bins = [0, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 25, 50, np.inf]

    heatmap_data = pd.concat(
        [
            pd.cut(
                admin_data["population_total"], population_bins, include_lowest=True
            ).rename("population"),
            pd.cut(admin_data["area_sq_km"], area_bins, include_lowest=True).rename(
                "area"
            ),
        ],
        axis=1,
    )
    heatmap_data = (
        heatmap_data.groupby(["population", "area"], observed=False)
        .size()
        .unstack()
        .sort_index(ascending=False)
    )
    count_total = heatmap_data.sum().sum()
    heatmap_data /= count_total

    return heatmap_data


def heatmap_with_hist_plot(
    data: pd.DataFrame,
    cmap: str | Colormap = "Greens",
    mask_value: Any = None,
    ax: Axes | None = None,
) -> tuple[Axes, Axes, Axes]:
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    y_data = data.sum(axis=1)[::-1]
    x_data = data.sum(axis=0)

    if mask_value is not None:
        mask = data == mask_value
    else:
        mask = np.zeros_like(data, dtype=bool)

    sns.heatmap(
        data,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.1,
        linecolor="grey",
        ax=ax,
        cbar=False,
    )
    ax.set_facecolor("lightgrey")
    ax.set_aspect(1, adjustable="box")

    hist_pad = 0.05
    hist_alpha = 0.75
    hist_bar_width = 0.97
    lim_adjust = 0.5
    hist_color = cmap(0.5) if sum(cmap(0.5)[:-1]) < 2.5 else cmap(1.0)  # noqa: PLR2004

    divider = make_axes_locatable(ax)

    xhist_ax = divider.append_axes("top", size=f"{100/len(y_data)}%", pad=hist_pad)
    x_data.plot.bar(  # type: ignore[call-overload]
        ax=xhist_ax, color=hist_color, alpha=hist_alpha, width=hist_bar_width
    )
    for i, v in enumerate(x_data):
        xhist_ax.text(i, 0.05, f"{v:.2f}", ha="center")
    xhist_ax.set_xlim(-lim_adjust, len(x_data) - lim_adjust)
    xhist_ax.set_ylim(0, 1)
    strip_axes(xhist_ax)
    xhist_ax.set_xlabel(None)

    yhist_ax = divider.append_axes("right", size=f"{100/len(x_data)}%", pad=hist_pad)
    y_data.plot.barh(  # type: ignore[call-overload]
        ax=yhist_ax, color=hist_color, alpha=hist_alpha, width=hist_bar_width
    )
    for i, v in enumerate(y_data):
        yhist_ax.text(0.05, i, f"{v:.2f}", ha="left", va="center")
    yhist_ax.set_ylim(-lim_adjust, len(y_data) - lim_adjust)
    yhist_ax.set_xlim(0, 1)
    strip_axes(yhist_ax)
    yhist_ax.set_ylabel(None)

    return ax, xhist_ax, yhist_ax


def plot_heatmaps_with_table(
    census_data: gpd.GeoDataFrame,
    summary_data: pd.DataFrame,
    iso3: str,
    year: str,
    save_path: str | Path | None = None,
) -> None:
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3, 1], wspace=0.3)

    admin_levels = sorted(census_data["admin_level"].unique())
    for i, admin_level in enumerate(admin_levels[-2:]):
        ax = fig.add_subplot(gs[(0, i)])
        heatmap_data = generate_heatmap_data(
            census_data[census_data["admin_level"] == admin_level]
        )
        hm_ax, top_ax, right_ax = heatmap_with_hist_plot(
            heatmap_data, cmap="Greens", mask_value=0.0, ax=ax
        )
        hm_ax.set_yticklabels(ax.get_yticklabels()[:-1] + ["0"])
        hm_ax.set_xlabel("Area (sq km)", fontsize=16)
        hm_ax.set_ylabel("Population", fontsize=16)
        top_ax.set_title(f"Admin Level {admin_level}", fontsize=18)

    ax_table = fig.add_subplot(gs[1, :])
    cell_text = summary_data.to_numpy().astype(str).tolist()
    col_labels = summary_data.columns.tolist()

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.scale(1.4, 2)
    ax_table.axis("off")
    table.set_fontsize(16)

    fig.suptitle(f"{year} {iso3} Census", fontsize=20, y=0.92)

    write_or_show(
        fig,
        save_path,
        format="pdf",
        dpi=300,
    )


#################
# Plotting task #
#################


def prepare_data_for_plotting(
    census_data: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    census_data["area_sq_km"] = census_data.to_crs(CRS.EQUAL_AREA).area / 10**6
    census_data["population_density"] = (
        census_data["population_total"] / census_data["area_sq_km"]
    )
    census_data["household_density"] = (
        census_data["household_total"] / census_data["area_sq_km"]
    )

    total_pop = census_data[census_data.admin_level == 0].population_total.iloc[0]

    census_data = census_data.to_crs(CRS.WGS84)
    summary_data = census_data.groupby("admin_level").agg(
        **{
            "Number of Units": ("shape_id", "count"),
            "Smallest Unit (sq. km.)": ("area_sq_km", "min"),
            "Largest Unit (sq. km.)": ("area_sq_km", "max"),
            "Average Unit Size (sq. km.)": ("area_sq_km", "mean"),
        }
    )

    summary_data["Average Population_Count Per Unit"] = (
        total_pop / summary_data["Number of Units"]
    )

    for col in summary_data:
        if col == "Number of Units":
            summary_data[col] = summary_data[col].astype(int)
        else:
            summary_data[col] = summary_data[col].apply(lambda x: f"{x:.2f}")

    return census_data, summary_data


def plot_census_summary_main(
    iso3: str,
    year: str,
    output_dir: str,
) -> None:
    """Make a census summary plot for a given iso3 and year.

    Parameters
    ----------
    iso3
        An iso3 code corresponding to a directory in the pop_data_dir.
    year
        A year corresponding to a directory in the iso3 subdirectory of the pop_data_dir.
    output_dir
        The directory where the final PDF will be saved.
    """
    pm_data = PopulationModelData(output_dir)
    census_data = pm_data.load_admin_training_data(iso3, year)
    census_data, summary_data = prepare_data_for_plotting(census_data)

    output_root = pm_data.census_qc

    admin_plot_path = output_root / f"{iso3}_{year}_admin_plot.pdf"
    plot_admin_level_data(census_data, iso3, year, save_path=admin_plot_path)
    heatmap_plot_path = output_root / f"{iso3}_{year}_heatmap_plot.pdf"
    plot_heatmaps_with_table(
        census_data, summary_data, iso3, year, save_path=heatmap_plot_path
    )


def subset_iso3_year_list(
    iso3_year_list: list[tuple[str, str]], iso3: str, year: str
) -> list[tuple[str, str]]:
    """Subset the list of iso3 and year tuples based on the input iso3 and year."""
    if iso3 == clio.RUN_ALL and year == clio.RUN_ALL:
        iso3_and_years = iso3_year_list
    elif iso3 == clio.RUN_ALL:
        iso3_and_years = [(i, y) for i, y in iso3_year_list if y == year]
        if not iso3_and_years:
            msg = f"Invalid year '{year}'."
            raise ValueError(msg)
    elif year == clio.RUN_ALL:
        iso3_and_years = [(i, y) for i, y in iso3_year_list if i == iso3]
        if not iso3_and_years:
            msg = f"Invalid iso3 '{iso3}'."
            raise ValueError(msg)
    else:
        if (iso3, year) not in iso3_year_list:
            msg = f"Invalid combination of iso3 '{iso3}' and year '{year}'."
            raise ValueError(msg)
        iso3_and_years = [(iso3, year)]
    return iso3_and_years


@click.command()  # type: ignore[arg-type]
@clio.with_iso3(allow_all=False)
@clio.with_year(allow_all=False)
@clio.with_output_directory(PEOPLE_PER_STRUCTURE_ROOT)
def plot_census_summary_task(
    iso3: str,
    year: str,
    output_dir: str,
) -> None:
    plot_census_summary_main(iso3, year, output_dir)


class PdfFileMerger(PdfMerger):
    """Super annoying that the real class isn't a context manager."""

    def __enter__(self) -> "PdfFileMerger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()


def merge_pdfs(pdf_root: str | Path, out_name: str) -> None:
    pdf_root = Path(pdf_root)
    output_path = pdf_root / out_name
    if output_path.exists():
        output_path.unlink()
    # Lexicographic sort gives us the order we want.
    pdfs = sorted(pdf_root.glob("*.pdf"))
    with PdfFileMerger() as merger:
        current_iso3 = None
        for current_page, input_path in enumerate(pdfs):
            iso3, _ = input_path.stem.split("_")[:2]
            merger.merge(current_page, str(input_path))

            # Bookmark it if it is the first page for a new iso3.
            if iso3 != current_iso3:
                merger.add_outline_item(iso3, current_page)
                current_iso3 = iso3

        touch(output_path, exist_ok=True)
        merger.write(str(output_path))

    for pdf in pdfs:
        pdf.unlink()


@click.command()  # type: ignore[arg-type]
@clio.with_iso3()
@clio.with_year()
@clio.with_output_directory(PEOPLE_PER_STRUCTURE_ROOT)
@clio.with_queue()
def plot_census_summary(
    iso3: str,
    year: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    iso3_year_list = pm_data.list_admin_training_data()
    iso3_year_list = subset_iso3_year_list(iso3_year_list, iso3, year)

    print(f"Launching {len(iso3_year_list)} plotting jobs.")
    jobmon.run_parallel(
        runner="ppstask preprocess",
        task_name="plot_census_summary",
        flat_node_args=(("iso3", "year"), iso3_year_list),
        task_args={
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "12G",
            "runtime": "5m",
            "project": "proj_rapidresponse",
        },
    )

    print("Merging PDFs.")
    merge_pdfs(pm_data.census_qc, "census_summary.pdf")
