import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterra as rt
import shapely
from rasterio.fill import fillnodata
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData

TIME_POINT = "2023q4"


def load_shape_population(pm_data: PopulationModelData, iso3: str) -> gpd.GeoDataFrame:
    pop = pm_data.load_raking_population("fhs_2021_wpp_2022")
    pop = pop[pop.year_id == int(TIME_POINT[:4])]
    shapes = pm_data.load_raking_shapes("fhs_2021_wpp_2022")

    all_pop = shapes.merge(pop, on="location_id")
    admin0_pop = all_pop[all_pop.ihme_loc_id == iso3]

    if admin0_pop.most_detailed.iloc[0] == 0:
        location_id = admin0_pop.location_id.to_numpy()[0]
        final_pop = all_pop[all_pop.parent_id == location_id]
    else:
        final_pop = admin0_pop
    return final_pop


def rake_itu_main(
    resolution: str,
    version: str,
    iso3: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    model_spec = pm_data.load_model_specification(resolution, version)
    modeling_frame = pm_data.load_modeling_frame(resolution)

    print("Building population shapefile")
    pop = load_shape_population(pm_data, iso3)

    itu_mask = pm_data.load_itu_mask(iso3)

    pop = pop.to_crs(itu_mask.crs)

    print("Building location id mask")
    location_mask = np.zeros_like(itu_mask.to_numpy())
    for location_id, geom in pop.set_index("location_id").geometry.to_dict().items():
        shape_mask, *_ = raster_geometry_mask(
            data_transform=itu_mask.transform,
            data_width=itu_mask.shape[1],
            data_height=itu_mask.shape[0],
            shapes=[geom],
            invert=True,
        )
        location_mask[shape_mask] = location_id

    # ITU has finer scale boundaries so we need to fill in the gaps
    # using the nearest location id
    to_fill_mask = (itu_mask.to_numpy() == 1) & (location_mask == 0)
    location_mask = fillnodata(location_mask, ~to_fill_mask, max_search_distance=100)

    print("Building modeling frame filters")
    block_key_x = modeling_frame["block_key"].apply(lambda x: int(x.split("X")[0][-4:]))
    boundary_blocks = 8
    near_antimeridian = (block_key_x <= boundary_blocks) | (
        block_key_x >= block_key_x.max() - boundary_blocks
    )
    block_filter = (
        near_antimeridian
        if str(itu_mask.crs) == pmc.CRSES["itu_anti_meridian"].code
        else pd.Series(data=True, index=modeling_frame.index)
    )

    xmin, xmax, ymin, ymax = itu_mask.bounds
    bbox = gpd.GeoSeries(
        [shapely.box(xmin, ymin, xmax, ymax)], crs=itu_mask.crs
    ).buffer(1000)
    isection_filter = (
        modeling_frame.to_crs(itu_mask.crs).buffer(0).intersects(bbox.union_all())
    )

    print("Getting intersecting blocks")
    blocks = (
        modeling_frame[block_filter & isection_filter]
        .dissolve("block_key")
        .to_crs(itu_mask.crs)
        .buffer(0)
    )

    print("Loading predictions")
    rasters = []
    for i, block_key in enumerate(blocks.index):
        print(f"Loading block {i}/{len(blocks)}: {block_key}")
        r = pm_data.load_raw_prediction(block_key, TIME_POINT, model_spec).resample_to(
            itu_mask, "sum"
        )
        rasters.append(r)
    prediction = rt.merge(rasters)

    print("Raking")
    prediction_array = prediction.to_numpy()
    raking_factor = np.ones_like(location_mask)
    for location_id, location_pop in (
        pop.set_index("location_id").wpp_population.to_dict().items()
    ):
        block_mask = location_mask == location_id
        pred_total = np.nansum(prediction_array[block_mask])
        raking_factor[block_mask] = location_pop / pred_total

    raked_arr = prediction_array * raking_factor  # type: ignore[operator]
    raked_arr[location_mask == 0] = np.nan

    raked_pop = rt.RasterArray(
        data=raked_arr,
        transform=itu_mask.transform,
        crs=itu_mask.crs,
        no_data_value=np.nan,
    )
    pm_data.save_raked_prediction(
        raked_pop, block_key=iso3, time_point=TIME_POINT, model_spec=model_spec
    )


@click.command()  # type: ignore[arg-type]
@clio.with_version()
@clio.with_resolution()
@clio.with_iso3()
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_itu_task(
    resolution: str,
    version: str,
    iso3: str,
    output_dir: str,
) -> None:
    rake_itu_main(
        resolution,
        version,
        iso3,
        output_dir,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_resolution()
@clio.with_version()
@clio.with_iso3(allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake_itu(
    resolution: str,
    version: str,
    iso3: str,
    output_dir: str,
    queue: str,
) -> None:
    """Rake populations to the ITU masks."""
    pm_data = PopulationModelData(output_dir)
    available_iso3s = pm_data.list_itu_iso3s()
    iso3s = clio.convert_choice(iso3, available_iso3s)

    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake_itu",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "50G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "iso3": iso3s,
        },
        task_args={
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        max_attempts=1,
        log_root=pm_data.log_dir("postprocess_rake_itu"),
    )
