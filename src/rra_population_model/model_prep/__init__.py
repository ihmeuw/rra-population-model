from rra_population_model.model_prep.features.runner import (
    features,
    features_task,
)
from rra_population_model.model_prep.modeling_frame.runner import (
    modeling_frame,
)

RUNNERS = {
    "modeling_frame": modeling_frame,
    "features": features,
}

TASK_RUNNERS = {
    "modeling_frame": modeling_frame,
    "features": features_task,
}
