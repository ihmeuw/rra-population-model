from rra_population_model.preprocess.census_data.runner import (
    census_data,
    census_data_task,
)
from rra_population_model.preprocess.plot_training_data import (
    plot_training_data,
    plot_training_data_task,
)
from rra_population_model.preprocess.raking_data.runner import (
    raking_data,
)
from rra_population_model.preprocess.summarize_training_data import (
    summarize_training_data,
    summarize_training_data_task,
)

RUNNERS = {
    "census_data": census_data,
    "raking_data": raking_data,
    "summarize_training_data": summarize_training_data,
    "plot_training_data": plot_training_data,
}

TASK_RUNNERS = {
    "census_data": census_data_task,
    "raking_data": raking_data,
    "summarize_training_data": summarize_training_data_task,
    "plot_training_data": plot_training_data_task,
}
