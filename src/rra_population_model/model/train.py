from pathlib import Path

import click
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from rra_population_model.data import (
    PopulationModelData,
)
from rra_population_model.model.modeling import (
    ModelSpecification,
    PPSDataModule,
    PPSModel,
)
from rra_population_model import cli_options as clio
from rra_population_model import constants as pmc


def train_main(resolution: str, model_name: str, output_root: str | Path) -> None:
    pm_data = PopulationModelData(output_root)
    model_spec = pm_data.load_model_specification(resolution, model_name)

    print("Setting up data module")
    data_module = PPSDataModule(model_specification=model_spec.model_dump())
    print("Loading data")
    data_module.setup()

    print("Setting up model")
    model = PPSModel(model_specification=model_spec.model_dump())

    print("Setting up trainer")
    trainer = Trainer(
        deterministic=True,
        log_every_n_steps=1,
        max_epochs=100000,
        callbacks=[EarlyStopping(monitor="val_loss")],
        default_root_dir=pm_data.model_root(resolution, model_name),
    )

    print("Training model")
    trainer.fit(model, data_module)

    print("Done!")


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(pmc.MODEL_ROOT)
def train(output_dir: str) -> None:
    provider = "ghsl"
    denominator = "residential_volume"
    use_ntl = "log"  # "yes", "no", "log"
    resolution = "100"

    suffix, ntl_feature = {
        "yes": ("ntl", ["nighttime_lights"]),
        "no": ("bd_only", []),
        "log": ("log_ntl", ["log_nighttime_lights"]),
    }[use_ntl]

    bd_features = [f"{provider}_{denominator}_{radius}m" for radius in pmc.FEATURE_AVERAGE_RADII] 


    name = f"{provider}_{denominator}_{suffix}"
    spec = ModelSpecification(
        name=name,
        denominator=f"{provider}_{denominator}",
        resolution=resolution,
        features=[*bd_features, *ntl_feature],
    )

    pm_data = PopulationModelData(output_dir)
    pm_data.save_model_specification(spec)
    train_main(resolution, name, output_dir)
