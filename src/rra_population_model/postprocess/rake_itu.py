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

RAKING_VERSION = "gbd_2023"


def load_admin_populations(
    pm_data: PopulationModelData,
    iso3: str,
    time_point: str,
) -> gpd.GeoDataFrame:
    raking_pop = pm_data.load_raking_population(version=RAKING_VERSION)
    all_pop = raking_pop.loc[raking_pop.most_detailed == 1].set_index(
        ["year_id", "location_id"]
    )["wpp_population"]

    # Interpolate the time point population
    if "q" in time_point:
        year, quarter = (int(s) for s in time_point.split("q"))
        if RAKING_VERSION == "gbd_2023":
            next_year = min(year + 1, 2024)
        else:
            next_year = min(year + 1, 2100)
        weight = (int(quarter) - 1) / 4

        prior_year_pop = all_pop.loc[year]
        next_year_pop = all_pop.loc[next_year]

        pop = (1 - weight) * prior_year_pop + weight * next_year_pop
    else:
        year = int(time_point)
        pop = all_pop.loc[year]

    admin_0 = raking_pop[raking_pop.ihme_loc_id == iso3]
    if admin_0.empty:
        raking_locs = (
            raking_pop[raking_pop.ihme_loc_id.str.startswith(iso3)]
            .location_id.unique()
            .tolist()
        )
        pop = pop.loc[raking_locs]
    else:
        pop = pop.loc[[admin_0.location_id.iloc[0]]]
    pop = pop.reset_index()

    admins = pm_data.load_raking_shapes(version=RAKING_VERSION)
    pop = admins[["location_id", "geometry"]].merge(pop, on="location_id")
    return pop


def rake_itu_main(
    resolution: str,
    version: str,
    iso3: str,
    time_point: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    model_spec = pm_data.load_model_specification(resolution, version)
    modeling_frame = pm_data.load_modeling_frame(resolution)

    print("Building population shapefile")
    pop = load_admin_populations(pm_data, iso3, time_point)

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
    prediction: rt.RasterArray | None = None
    for i, block_key in enumerate(blocks.index):
        print(f"Loading block {i}/{len(blocks)}: {block_key}")
        r = pm_data.load_raw_prediction(block_key, time_point, model_spec).resample_to(
            itu_mask, "sum"
        )
        if prediction is None:
            prediction = r
        else:
            if np.nansum(r) == 0:
                continue
            prediction = rt.merge([prediction, r])

    print("Raking")
    assert isinstance(prediction, rt.RasterArray)  # noqa: S101
    prediction_array = prediction.to_numpy()
    raking_factor = np.ones_like(location_mask)
    for location_id, location_pop in (
        pop.set_index("location_id").wpp_population.to_dict().items()
    ):
        block_mask = location_mask == location_id
        pred_total = np.nansum(prediction_array[block_mask])
        raking_factor[block_mask] = location_pop / pred_total

    raked_arr = prediction_array * raking_factor
    raked_arr[location_mask == 0] = np.nan

    raked_pop = rt.RasterArray(
        data=raked_arr,
        transform=itu_mask.transform,
        crs=itu_mask.crs,
        no_data_value=np.nan,
    )
    pm_data.save_raked_prediction(
        raked_pop, block_key=iso3, time_point=time_point, model_spec=model_spec
    )


@click.command()
@clio.with_version()
@clio.with_resolution()
@clio.with_iso3()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_itu_task(
    resolution: str,
    version: str,
    iso3: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_itu_main(
        resolution,
        version,
        iso3,
        time_point,
        output_dir,
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_copy_from_version()
@clio.with_iso3(allow_all=True)
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake_itu(
    resolution: str,
    version: str,
    copy_from_version: str | None,
    iso3: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    """Rake populations to the ITU masks."""
    pm_data = PopulationModelData(output_dir)
    pm_data.maybe_copy_version(resolution, version, copy_from_version)

    available_iso3s = pm_data.list_itu_iso3s()
    iso3s = clio.convert_choice(iso3, available_iso3s)

    prediction_time_points = pm_data.list_raw_prediction_time_points(
        resolution, version
    )
    time_points = clio.convert_choice(time_point, prediction_time_points)

    print(f"Launching {len(iso3s) * len(time_points)} tasks")

    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake_itu",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "75G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "iso3": iso3s,
            "time-point": time_points,
        },
        task_args={
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("postprocess_rake_itu"),
    )
