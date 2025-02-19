from rra_population_model.model_prep.features.runner import (
    features,
    features_task,
)
from rra_population_model.model_prep.modeling_frame.runner import (
    modeling_frame,
)
from rra_population_model.model_prep.training_data.runner import (
    training_data,
    training_data_task,
)

RUNNERS = {
    "modeling_frame": modeling_frame,
    "features": features,
    "training_data": training_data,
}

TASK_RUNNERS = {
    "modeling_frame": modeling_frame,
    "features": features_task,
    "training_data": training_data_task,
}
