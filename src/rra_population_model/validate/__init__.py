from rra_population_model.validate.metrics.runner import (
    metrics,
    pixel_metrics_task,
)
from rra_population_model.validate.comparison import (
    comparison_validation,
    comparison_validation_task,
)
from rra_population_model.validate.diagnostics import (
    diagnostics,
    diagnostics_attribute_task,
    diagnostics_census_task,
    diagnostics_collate_task,
    diagnostics_scan_task,
)

RUNNERS = {
    "metrics": metrics,
    "comparison": comparison_validation,
    "diagnostics": diagnostics,
}

TASK_RUNNERS = {
    "pixel_metrics": pixel_metrics_task,
    "comparison_validation": comparison_validation_task,
    "diagnostics_census": diagnostics_census_task,
    "diagnostics_scan": diagnostics_scan_task,
    "diagnostics_attribute": diagnostics_attribute_task,
    "diagnostics_collate": diagnostics_collate_task,
}
