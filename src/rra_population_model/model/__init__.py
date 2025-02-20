from rra_population_model.model.inference import (
    inference,
    inference_task,
)
from rra_population_model.model.train.runner import (
    train,
    train_task,
)

RUNNERS = {
    "inference": inference,
    "train": train,
}

TASK_RUNNERS = {
    "inference": inference_task,
    "train": train_task,
}
