import click
import numpy as np
import rasterra as rt
from rasterra._features import raster_geometry_mask
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import PopulationModelData
from rra_population_model.postprocess.utils import get_prediction_time_point


def rake_main(
    resolution: str,
    version: str,
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
    unraked_data = pm_data.load_raw_prediction(
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

    print("Saving raked prediction")
    pm_data.save_raked_prediction(raked, block_key, time_point, model_spec)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_block_key()
@clio.with_time_point(choices=None)
@clio.with_output_directory(pmc.MODEL_ROOT)
def rake_task(
    resolution: str,
    version: str,
    block_key: str,
    time_point: str,
    output_dir: str,
) -> None:
    rake_main(resolution, version, block_key, time_point, output_dir)


@click.command()
@clio.with_resolution(allow_all=False)
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def rake(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

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
            "version": [version],  # [f"2025_06_21.00{x}" for x in range(1, 5)],
            "block-key": block_keys,
            "time-point": time_points,
        },
        task_args={
            # "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
        },
        max_attempts=3,
        log_root=pm_data.log_dir("postprocess_rake"),
    )
