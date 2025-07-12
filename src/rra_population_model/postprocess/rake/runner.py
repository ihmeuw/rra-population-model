import click
import numpy as np
import rasterra as rt
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon
import tqdm

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import get_prediction_time_point


def rake_main(
    resolution: str,
    version: str,
    input_data: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    print("Loading metadata")
    model_spec = pm_data.load_model_specification(resolution, version)
    prediction_time_point = get_prediction_time_point(
        pm_data, resolution, version, time_point
    )
    print("Loading unraked prediction")
    if input_data == 'raw':
        unraked_data = pm_data.load_raw_prediction(
            block_key, prediction_time_point, model_spec
        )
    elif input_data == 'raked':
        unraked_data = pm_data.load_raked_prediction(
            block_key, prediction_time_point, model_spec
        )

    print("Loading raking factors")
    raking_data = pm_data.load_raking_factors(
        time_point,
        model_spec,
        filters=[("block_key", "==", block_key)],
    )

    print("Raking")
    if raking_data.empty:
        raking_factor = rt.RasterArray(
            np.nan * np.ones_like(unraked_data),
            transform=unraked_data.transform,
            crs=unraked_data.crs,
            no_data_value=np.nan,
        )
        raked = raking_factor
    else:
        raking_factor_data = np.nan * np.ones_like(unraked_data)
        for geom, rf in raking_data[["geometry", "raking_factor"]].itertuples(
            index=False
        ):
            shape_mask, *_ = raster_geometry_mask(
                data_transform=unraked_data.transform,
                data_width=unraked_data.shape[1],
                data_height=unraked_data.shape[0],
                shapes=[geom],
                invert=True,
            )
            raking_factor_data[shape_mask] = rf

        raking_factor = rt.RasterArray(
            raking_factor_data,
            transform=unraked_data.transform,
            crs=unraked_data.crs,
            no_data_value=np.nan,
        )
        raked = unraked_data * raking_factor

        if input_data == 'raw':
            print("Loading inference data")
            model_frame = pm_data.load_modeling_frame(resolution)
            model_frame = model_frame.loc[model_frame['block_key'] == block_key]
            census_population = []
            census_weight = []
            for tile_key in tqdm.tqdm(model_frame['tile_key'].to_list()):
                tile_census_population = pm_data.load_tile_inference_data(
                    resolution,
                    tile_key,
                    time_point,
                    f'population_{model_spec.denominator}',
                )
                if tile_census_population is None:
                    tile_census_population = pm_data.load_feature(
                        resolution=resolution,
                        block_key=block_key,
                        feature_name=model_spec.denominator,
                        time_point=time_point,
                        subset_bounds=model_frame[model_frame.tile_key == tile_key].geometry.iloc[0],
                    )
                    tile_census_population = rt.RasterArray(
                        np.zeros_like(tile_census_population),
                        transform=tile_census_population.transform,
                        crs=tile_census_population.crs,
                        no_data_value=np.nan,
                    )
                    tile_census_weight = rt.RasterArray(
                        np.zeros_like(tile_census_population),
                        transform=tile_census_population.transform,
                        crs=tile_census_population.crs,
                        no_data_value=np.nan,
                    )
                else:
                    tile_census_weight = pm_data.load_tile_inference_data(
                        resolution,
                        tile_key,
                        time_point,
                        'area_weight',
                    )
                census_population.append(tile_census_population)
                census_weight.append(tile_census_weight)

            census_population = rt.merge(census_population)
            if np.nansum(census_population) > 0 and np.nansum(raked) > 0:
                array = census_population.to_numpy()
                nan_mask = np.isnan(array)
                array[nan_mask] = 0
                census_population = rt.RasterArray(
                    array,
                    transform=raked.transform,
                    crs=raked.crs,
                    no_data_value=np.nan,
                )

                census_weight = rt.merge(census_weight)
                array = census_weight.to_numpy()
                nan_mask = np.isnan(array)
                array[nan_mask] = 0
                census_weight = rt.RasterArray(
                    array,
                    transform=raked.transform,
                    crs=raked.crs,
                    no_data_value=np.nan,
                )

                print("Splicing in inference data")
                raked = (
                    (raked * (1 - census_weight))
                    + (census_population * census_weight)
                )
            else:
                print("No population to splice")

    print("Saving raked prediction")
    pm_data.save_raked_prediction(raked, block_key, time_point, model_spec)


@click.command()
@clio.with_resolution()
@clio.with_version()
@click.option("--input-data", type=str, required=True)
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_task(
    resolution: str,
    version: str,
    input_data: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_main(resolution, version, input_data, block_key, time_point, output_dir)


@click.command()
@clio.with_resolution(allow_all=False)
@clio.with_version()
@click.option("--input-data", type=str, required=True)
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake(
    resolution: str,
    version: str,
    input_data: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    if input_data == 'raw':
        if len(list(pm_data.raked_predictions_root(resolution, version).iterdir())) > 0:
            raise ValueError(f'Raked predictions already exist, cannot run with `input_data` set to `raw`.')
    elif input_data != 'raked':
        raise ValueError(f'Invalid `input_data` type: {input_data}')

    rf_time_points = pm_data.list_raking_factor_time_points(resolution, version)

    time_points = clio.convert_choice(time_point, rf_time_points)

    model_frame = pm_data.load_modeling_frame(resolution)
    block_keys = model_frame.block_key.unique().tolist()

    print(f"Raking {len(block_keys) * len(time_points)} blocks")
    jobmon.run_parallel(
        runner="pmtask postprocess",
        task_name="rake",
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "50G",
            "runtime": "25m",
            "project": "proj_rapidresponse",
        },
        node_args={
            # "version": [f"2025_06_21.00{x}" for x in range(1, 5)],
            "block-key": block_keys,
            "time-point": time_points,
        },
        task_args={
            "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
            "input-data": input_data,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("postprocess_rake"),
    )
