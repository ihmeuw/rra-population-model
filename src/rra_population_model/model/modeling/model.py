"""Generic model class for people per structure modeling."""

from typing import Any

import lightning
import rasterra as rt
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Adam, Optimizer

from rra_population_model.model.modeling.datamodel import (
    Metric,
    ModelSpecification,
    ModelTarget,
    ScoreFunction,
)
from rra_population_model.model.modeling.metrics import (
    get_metric_function,
)
from rra_population_model.model.modeling.transformations import (
    get_transformation,
)


class PPSModel(lightning.LightningModule):
    """Generic model class for people per structure modeling."""

    def __init__(
        self,
        model_specification: dict[str, Any],
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.specification = ModelSpecification(**model_specification)

        lightning.seed_everything(self.specification.random_seed, workers=True)
        self.model = self._build_model(self.specification)

        self.training_target = self.specification.training_target
        self.training_level = self.training_target.split("_")[0]
        self.loss_target = self.specification.loss_target
        self.loss_function = self._build_score_function(
            model_target=self.training_target,
            scoring_target=self.loss_target,
            scoring_metric=self.specification.loss_metric,
        )

        self.test_target = self.specification.test_target
        self.test_level = self.test_target.split("_")[0]
        self.evaluation_target = self.specification.evaluation_target
        self.evaluation_function = self._build_score_function(
            model_target=self.test_target,
            scoring_target=self.evaluation_target,
            scoring_metric=self.specification.evaluation_metric,
        )

        self.epoch = 0
        self.verbose = verbose

    ###################
    # Lightning hooks #
    ###################

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        x = batch[f"{self.training_level}_features"]
        model_prediction = self.model(x)
        y_true = batch[str(self.loss_target)]
        loss = self.loss_function(
            batch["pixel_built"],
            batch["admin_built"],
            batch["pixel_area_weights"],
            model_prediction,
            y_true,
        )
        self.log("train_loss", loss, prog_bar=self.verbose)
        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        x = batch[f"{self.test_level}_features"]
        model_prediction = self.model(x)
        y_true = batch[str(self.test_target)]
        loss = self.evaluation_function(
            batch["pixel_built"],
            batch["admin_built"],
            batch["pixel_area_weights"],
            model_prediction,
            y_true,
        )
        self.log("val_loss", loss, prog_bar=self.verbose)
        return loss

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        x = batch[f"{self.test_level}_features"]
        model_prediction = self.model(x)
        y_true = batch[str(self.test_target)]
        loss = self.evaluation_function(
            batch["pixel_built"],
            batch["admin_built"],
            batch["pixel_area_weights"],
            model_prediction,
            y_true,
        )
        self.log("test_loss", loss)
        return loss

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> STEP_OUTPUT:
        x = batch["features"]
        built = batch["built"]
        pred = self.model(x).exp().squeeze(-1) * built
        batch["population_raster"] = []
        for i, template in enumerate(batch["raster_template"]):
            batch["population_raster"].append(
                rt.RasterArray(
                    pred[i].numpy(),
                    transform=template.transform,
                    crs=template.crs,
                    no_data_value=template.no_data_value,
                )
            )
        return batch

    def configure_optimizers(self) -> Optimizer:
        if self.specification.optimizer == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.specification.learning_rate,
            )
        else:
            msg = f"Optimizer {self.specification.optimizer} not implemented"
            raise NotImplementedError(msg)
        return optimizer

    #################
    # Model helpers #
    #################

    @staticmethod
    def _build_model(specification: ModelSpecification) -> torch.nn.Module:
        """Build the model."""
        model_type = specification.mtype
        if model_type == "linear":
            return torch.nn.Linear(len(specification.features), 1)
        else:
            msg = f"Model type {model_type} not implemented"
            raise NotImplementedError(msg)

    @staticmethod
    def _build_score_function(
        model_target: ModelTarget,
        scoring_target: ModelTarget,
        scoring_metric: Metric,
    ) -> ScoreFunction:
        """Get a scoring function for the model."""

        def scoring_function(
            pixel_building_density: torch.Tensor,
            admin_building_density: torch.Tensor,
            pixel_area_weights: torch.Tensor,
            model_prediction: torch.Tensor,
            y_true: torch.Tensor,
        ) -> torch.Tensor:
            # Transform the model output to the loss target
            transformation = get_transformation(
                model_target,
                scoring_target,
                pixel_building_density,
                admin_building_density,
                pixel_area_weights,
            )
            y_pred = transformation(model_prediction)
            metric_function = get_metric_function(scoring_metric)
            return metric_function(y_pred, y_true)

        return scoring_function
