"""Metrics for evaluating model performance."""

import torch

from rra_population_model.model.modeling.datamodel import (
    Metric,
    MetricFunction,
)


def get_metric_function(metric: Metric) -> MetricFunction:
    """Get the metric function for the given metric."""
    return {
        Metric.MAE: mae,
        Metric.MSE: mse,
        Metric.RMSE: rmse,
        Metric.NRMSD: nrmsd,
        Metric.MAPE: mape,
        Metric.SMAPE: smape,
        Metric.R2: r2,
    }[metric]


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return (y_true - y_pred).abs().mean()


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return ((y_true - y_pred) ** 2).mean()  # type: ignore[no-any-return]


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    return mse(y_true, y_pred).sqrt()


def nrmsd(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Normalized root mean squared error."""
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())


def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute percentage error."""
    return (torch.abs(y_true - y_pred) / y_true).mean()


def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Symmetric mean absolute percentage error."""
    return torch.mean(
        2 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred))
    )


def r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination."""
    rss = torch.sum((y_true - y_pred) ** 2)
    tss = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - rss / tss  # type: ignore[no-any-return]
