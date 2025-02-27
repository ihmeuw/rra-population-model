from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import click
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.model.modeling import (
    InferenceDataModule,
    ModelSpecification,
)


class CustomWriter(BasePredictionWriter):
    def __init__(
        self,
        pm_data: PopulationModelData,
        model_spec: ModelSpecification,
        time_point: str,
        write_interval: Literal["batch", "epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self.pm_data = pm_data
        self.model_spec = model_spec
        self.time_point = time_point

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: dict[str, Any],
        batch_indices: Sequence[int] | None,  # noqa: ARG002
        batch: dict[str, Any],  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        for i, block in enumerate(outputs["block"]):
            block_key = block.split("_")[0]
            raster = outputs["population_raster"][i]
            self.pm_data.save_raw_prediction(
                raster,
                block_key,
                self.time_point,
                self.model_spec,
            )


def inference_main(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str | Path,
    progress_bar: bool,
) -> None:
    pm_data = PopulationModelData(output_dir)

    model = pm_data.load_model(resolution, version)
    model_spec = model.specification

    modeling_frame = pm_data.load_modeling_frame(model_spec.resolution)
    block_keys = modeling_frame.block_key.unique().tolist()
    block_keys = [
        block_key
        for block_key in block_keys
        if not pm_data.raw_prediction_path(block_key, time_point, model_spec).exists()
    ]

    datamodule = InferenceDataModule(
        model_spec.model_dump(),
        block_keys,
        time_point,
        num_workers=4,
    )
    pred_writer = CustomWriter(
        pm_data, model.specification, time_point, write_interval="batch"
    )
    trainer = Trainer(
        callbacks=[pred_writer],
        enable_progress_bar=progress_bar,
        devices=2,
    )
    trainer.predict(model, datamodule, return_predictions=False)


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_progress_bar()
def inference_task(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    progress_bar: bool,
) -> None:
    inference_main(
        resolution,
        version,
        time_point,
        output_dir,
        progress_bar,
    )


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_time_point(choices=None, allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def inference(
    resolution: str,
    version: str,
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    feature_time_points = pm_data.list_feature_time_points(resolution)
    time_points = clio.convert_choice(time_point, feature_time_points)
    print(f"Running inference for {len(time_points)} time points.")

    jobmon.run_parallel(
        runner="pmtask model",
        task_name="inference",
        node_args={
            "time-point": time_points,
        },
        task_args={
            "resolution": resolution,
            "version": version,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "memory": "20G",
            "runtime": "480m",
            "project": "proj_rapidresponse",
        },
        log_root=pm_data.log_dir("model_inference"),
    )
