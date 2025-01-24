from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import click
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from rra_tools import jobmon

from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.model.modeling import (
    InferenceDataModule,
    PPSModel,
    ModelSpecification,
)

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc


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
            self.pm_data.save_prediction(
                raster,
                block_key,
                self.time_point,
                self.model_spec,
            )


def inference_main(
    model_path: str | Path,
    time_point: str,
    output_dir: str | Path,
    progress_bar: bool,
) -> None:
    pm_data = PopulationModelData(output_dir)
    model = PPSModel.load_from_checkpoint(model_path)
    model_spec = model.specification

    modeling_frame = pm_data.load_modeling_frame(model_spec.resolution)
    block_keys = modeling_frame.block_key.unique().tolist()

    datamodule = InferenceDataModule(
        model.specification.model_dump(),
        block_keys,
        time_point,
    )
    pred_writer = CustomWriter(pm_data, model_spec, time_point, write_interval="batch")
    trainer = Trainer(
        callbacks=[pred_writer],
        enable_progress_bar=progress_bar,
    )
    trainer.predict(model, datamodule, return_predictions=False)


@click.command()  # type: ignore[arg-type]
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the model checkpoint.",
)
@clio.with_time_point()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_progress_bar()
def inference_task(
    model_path: str,
    time_point: str,
    output_dir: str,
    progress_bar: bool,
) -> None:
    inference_main(
        model_path,
        time_point,
        output_dir,
        progress_bar,
    )


@click.command()  # type: ignore[arg-type]
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the model checkpoint.",
)
@clio.with_time_point(allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def inference(
    model_path: str,
    time_point: list[str],
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)

    jobmon.run_parallel(
        runner="pmtask model",
        task_name="inference",
        node_args={
            "time-point": time_point,
        },
        task_args={
            "model-path": model_path,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "memory": "20G",
            "runtime": "480m",
            "project": "proj_rapidresponse",
        },
        log_root=pm_data.predictions,
    )
