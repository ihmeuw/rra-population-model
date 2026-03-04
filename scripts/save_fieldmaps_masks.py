import tqdm

import pandas as pd
import numpy as np

import geopandas as gpd
import rasterra as rt
import contextily as ctx
import rasterio as rio

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rra_population_model.constants as pmc
from rra_population_model.data import PopulationModelData, save_raster

ITU_MASKS_ROOT = pmc.MODEL_ROOT / 'admin-inputs' / 'itu-masks'
FIELDMAPS_FILES = {
    'humanitarian': 'adm1_polygons_UNCOD_geoBoundaries.parquet',
    'open': 'adm1_polygons_geoBoundaries.parquet',
}


def load_data(dataset_name: str, pm_data: PopulationModelData):
    itu_caribbean = pd.read_csv(ITU_MASKS_ROOT / 'datasets' / 'itu_caribbean.csv')

    data = gpd.read_parquet(ITU_MASKS_ROOT / 'datasets' / FIELDMAPS_FILES[dataset_name])

    data_caribbean = data.loc[data['iso_2'].isin(itu_caribbean['Iso2Code'])]
    data_caribbean = data_caribbean.loc[
        # multiple rows for Jamaica, just keep matching admin0
        (data_caribbean['iso_2'] != 'JM') | (data_caribbean['adm0_name'] == 'Jamaica')
    ]
    data_caribbean = data_caribbean.loc[:, ['adm0_name', 'iso_2', 'iso_3', 'geometry']].dissolve(by=['adm0_name', 'iso_2', 'iso_3'])

    missing = [i for i in itu_caribbean['Iso2Code'] if i not in data_caribbean.index.get_level_values('iso_2')]
    if missing:
        raise ValueError(f"Missing the following countries: {','.join(missing)}")

    itu_iso3s = pm_data.list_itu_iso3s()
    if "RWA" not in itu_iso3s:
        itu_iso3s += ["RWA"]
    caribbean_iso3s = data_caribbean.index.get_level_values('iso_3').to_list()
    itu_iso3s = [i for i in itu_iso3s if i not in caribbean_iso3s]

    data_other = data.loc[data['iso_3'].isin(itu_iso3s)].loc[:, ['adm0_name', 'iso_2', 'iso_3', 'geometry']].dissolve(by=['adm0_name', 'iso_2', 'iso_3'])

    data = pd.concat([data_other, data_caribbean])

    return data


def polygon_to_raster_mask(iso3, gdf, resolution, pm_data):
    """
    Creates a binary raster mask from a polygon with a specific resolution.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the polygon(s).
        resolution (int): The desired resolution in meters.
        output_file (str): The path for the output GeoTIFF file.
    """
    if iso3 in pm_data.list_itu_iso3s():
        # Use existing mask as template for resolution and extent
        template_mask = pm_data.load_itu_mask(iso3)
        target_crs = template_mask.crs
    else:
        # Default to Web Mercator and a reasonable resolution
        template_mask = None
        target_crs = "EPSG:3857"

    gdf = gdf.loc[:, :, iso3].to_crs(target_crs)

    # Ensure the GeoDataFrame has a projected CRS with meters as units
    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("Input GeoDataFrame must have a projected CRS (like UTM).")

    # Get the bounding box of the polygon(s)
    minx, miny, maxx, maxy = (float(bound) for bound in gdf.total_bounds)

    # Calculate raster dimensions based on resolution
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # Define the georeferencing transform
    transform = rio.transform.from_bounds(minx, miny, minx + width * resolution, miny + height * resolution, width, height)

    # Get shapes and values for rasterization
    shapes = ((geom, 1.0) for geom in gdf.geometry)

    # Create the binary mask as a NumPy array
    binary_mask = rio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype=np.float32
    )

    raster_mask = rt.RasterArray(
        data=binary_mask,
        transform=transform,
        crs=target_crs,
        no_data_value=0.0,
    )

    return template_mask, raster_mask


def main():
    pm_data = PopulationModelData()

    data_humanitarian = load_data("humanitarian", pm_data)

    with PdfPages(ITU_MASKS_ROOT / "datasets" / "mask_update.pdf") as pdf:
        for iso3 in tqdm.tqdm(data_humanitarian.index.get_level_values('iso_3'), total=len(data_humanitarian)):
            template_mask, raster_mask = polygon_to_raster_mask(
                iso3=iso3,
                gdf=data_humanitarian,
                resolution=100,
                pm_data=pm_data,
            )
            save_raster(raster_mask, ITU_MASKS_ROOT / f"{iso3}.tif")

            if template_mask is not None:
                diff = rt.RasterArray(
                    (raster_mask.resample_to(template_mask).to_numpy() - template_mask.to_numpy()),
                    transform=template_mask.transform,
                    crs=template_mask.crs,
                    no_data_value=0.0
                )

                fig, ax = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)
                template_mask.plot(ax=ax[0], alpha=0.8)
                raster_mask.plot(ax=ax[1], alpha=0.8)
                diff.plot(ax=ax[2], alpha=0.8)
                if raster_mask.crs != "EPSG:3832":
                    ctx.add_basemap(ax[0], crs=raster_mask.crs, source=ctx.providers.Esri.WorldImagery, alpha=0.6, attribution=False)
                    ctx.add_basemap(ax[1], crs=raster_mask.crs, source=ctx.providers.Esri.WorldImagery, alpha=0.6, attribution=False)
                    ctx.add_basemap(ax[2], crs=raster_mask.crs, source=ctx.providers.Esri.WorldImagery, alpha=0.6, attribution=False)
                ax[0].set_title('Old mask')
                ax[1].set_title(
                    f'{iso3}'
                    '\nNew mask'
                )
                ax[2].set_title('Diff')
                fig.tight_layout()
                pdf.savefig(fig)  # saves the current figure into a pdf page
                plt.close()


if __name__ == "__main__":
    main()
