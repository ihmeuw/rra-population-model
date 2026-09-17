from rra_population_model.validate.metrics.runner import (
    metrics,
    pixel_metrics_task,
)
from rra_population_model.validate.comparison import (
    comparison_validation,
    comparison_validation_task,
)

RUNNERS = {
    "metrics": metrics,
    "comparison": comparison_validation,
}

TASK_RUNNERS = {
    "pixel_metrics": pixel_metrics_task,
    "comparison_validation": comparison_validation_task,
}
