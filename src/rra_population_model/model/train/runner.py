import itertools

import click
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from rra_tools import jobmon

from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc
from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.model.modeling import (
    ModelSpecification,
    PPSDataModule,
    PPSModel,
)
from rra_population_model.model.train import utils


def train_main(
    resolution: str,
    version: str,
    denominator: str,
    ntl_option: str,
    output_root: str,
    *,
    verbose: bool = False,
) -> None:
    pm_data = PopulationModelData(output_root)
    # First generate the model specification and mint a new version directory
    print("Setting up model specification")
    version_root = pm_data.model_version_root(resolution, version)
    ntl_feature = {
        "none": [],
        "ntl": ["nighttime_lights"],
        "log_ntl": ["log_nighttime_lights"],
    }[ntl_option]
    bd_features: list[str] = []
    model_spec = ModelSpecification(
        model_version=version,
        model_root=str(pm_data.root),
        output_root=str(version_root),
        denominator=denominator,
        resolution=resolution,
        features=[*bd_features, *ntl_feature],
    )
    pm_data.save_model_specification(model_spec)

    print("Setting up data module")
    data_module = PPSDataModule(
        model_specification=model_spec.model_dump(), verbose=verbose
    )
    print("Loading data")
    data_module.setup()

    print("Setting up model")
    model = PPSModel(model_specification=model_spec.model_dump(), verbose=verbose)

    print("Setting up trainer")
    trainer = Trainer(
        deterministic=True,
        log_every_n_steps=1,
        max_epochs=100000,
        callbacks=[EarlyStopping(monitor="val_loss")],
        default_root_dir=version_root,
    )

    print("Training model")
    trainer.fit(model, data_module)

    ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    ckpt_link = version_root / "best_model.ckpt"
    ckpt_link.symlink_to(ckpt_path)

    print("Done!")


@click.command()
@clio.with_resolution()
@clio.with_version()
@clio.with_denominator()
@clio.with_ntl_option()
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_verbose()
def train_task(
    resolution: str,
    version: str,
    denominator: str,
    ntl_option: str,
    output_dir: str,
    verbose: bool,
) -> None:
    train_main(
        resolution, version, denominator, ntl_option, output_dir, verbose=verbose
    )


@click.command()
@clio.with_resolution()
@clio.with_denominator(allow_all=True)
@clio.with_ntl_option(allow_all=True)
@clio.with_output_directory(pmc.MODEL_ROOT)
@clio.with_queue()
def train(
    resolution: str,
    denominator: list[str],
    ntl_option: list[str],
    output_dir: str,
    queue: str,
) -> None:
    pm_data = PopulationModelData(output_dir)
    today, last_version = utils.get_last_run_version(pm_data.model_root(resolution))
    node_args = []
    for i, (denom, ntl) in enumerate(itertools.product(denominator, ntl_option)):
        version = f"{today}.{last_version + i + 1:03d}"
        print(f"{version}: {denom} {ntl}")
        node_args.append((version, denom, ntl))

    jobmon.run_parallel(
        runner="pmtask model",
        task_name="train",
        flat_node_args=(
            ("version", "denominator", "ntl-option"),
            node_args,
        ),
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "240G",
            "runtime": "150m",
            "project": "proj_rapidresponse",
        },
        log_root=pm_data.log_dir("model_train"),
        max_attempts=1,
    )
