from rra_population_model.validate.metrics.runner import (
    metrics,
    pixel_metrics_task,
)

RUNNERS = {
    "metrics": metrics,
}

TASK_RUNNERS = {
    "pixel_metrics": pixel_metrics_task,
}
