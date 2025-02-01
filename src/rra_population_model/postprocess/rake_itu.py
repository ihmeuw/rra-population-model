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
from rra_population_model.data import PopulationModelData, RRAPopulationData


def load_shape_population(pop_data: RRAPopulationData, iso3: str) -> gpd.GeoDataFrame:
    pop = pop_data.load_ihme_populations()
    h = pop_data.load_ihme_hierarchy()
    shps = pop_data.load_ihme_shapes()

    pop = pop[pop.year_id == pop.year_id.max()]
    pop = pop.merge(h[["location_id", "local_id"]], on="location_id")
    pop = shps[["location_id", "geometry"]].merge(pop, on="location_id")
    admin0_pop = pop[pop.local_id == iso3]
    if admin0_pop.most_detailed.iloc[0] == 0:
        location_id = admin0_pop.location_id.to_numpy()[0]
        pop = pop[pop.parent_id == location_id]
    else:
        pop = admin0_pop
    return pop


def rake_itu_main(
    iso3: str, model_name: str, resolution: str, pop_data_dir: str, output_dir: str
) -> None:
    pop_data = RRAPopulationData(pop_data_dir)
    pm_data = PopulationModelData(output_dir)

    model_spec = pm_data.load_model_specification(resolution, model_name)

    print("Building population shapefile")
    pop = load_shape_population(pop_data, iso3)

    modeling_frame = pm_data.load_modeling_frame(resolution)
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
        r = pm_data.load_prediction(block_key, "2023q4", model_spec).resample_to(
            itu_mask, "sum"
        )
        rasters.append(r)
    prediction = rt.merge(rasters)

    print("Raking")
    prediction_array = prediction.to_numpy()
    raking_factor = np.ones_like(location_mask)
    for location_id, location_pop in (
        pop.set_index("location_id").population.to_dict().items()
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
    pm_data.save_itu_results(raked_pop, iso3, model_spec)


@click.command()  # type: ignore[arg-type]
@clio.with_iso3()
@click.option(
    "--model-name",
    required=True,
    type=click.STRING,
    help="Name of the model to rake.",
)
@clio.with_resolution()
@clio.with_input_directory("pop-data", pmc.POPULATION_DATA_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_itu_task(
    iso3: str,
    model_name: str,
    resolution: str,
    pop_data_dir: str,
    output_dir: str,
) -> None:
    rake_itu_main(iso3, model_name, resolution, pop_data_dir, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_iso3(allow_all=True)
@click.option(
    "--model-name",
    required=True,
    type=click.STRING,
    help="Name of the model to rake.",
)
@clio.with_resolution()
@clio.with_input_directory("pop-data", pmc.POPULATION_DATA_ROOT)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake_itu(
    iso3: str,
    model_name: str,
    resolution: str,
    pop_data_dir: str,
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
            "model-name": model_name,
            "resolution": resolution,
            "pop-data-dir": pop_data_dir,
            "output-dir": output_dir,
        },
        max_attempts=3,
        log_root=pm_data.itu_results,
    )
