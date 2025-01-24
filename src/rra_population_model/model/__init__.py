from rra_population_model.model.inference import (
    inference,
    inference_task,
)
from rra_population_model.model.train import (
    train,
)

RUNNERS = {
    "train": train,
    "inference": inference,
}

TASK_RUNNERS = {
    "train": train,
    "inference": inference_task,
}
