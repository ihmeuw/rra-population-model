from rra_population_model.preprocess.census_data.runner import (
    census_data,
    census_data_task,
)
from rra_population_model.preprocess.features.runner import (
    features,
    features_task,
)
from rra_population_model.preprocess.modeling_frame import (
    modeling_frame,
)
from rra_population_model.preprocess.plot_census import (
    plot_census_summary,
    plot_census_summary_task,
)
from rra_population_model.preprocess.plot_training_data import (
    plot_training_data,
    plot_training_data_task,
)
from rra_population_model.preprocess.summarize_training_data import (
    summarize_training_data,
    summarize_training_data_task,
)
from rra_population_model.preprocess.training_data import (
    training_data,
    training_data_task,
)

RUNNERS = {
    "plot_census_summary": plot_census_summary,
    "modeling_frame": modeling_frame,
    "features": features,
    "census_data": census_data,
    "training_data": training_data,
    "summarize_training_data": summarize_training_data,
    "plot_training_data": plot_training_data,
}

TASK_RUNNERS = {
    "plot_census_summary": plot_census_summary_task,
    "modeling_frame": modeling_frame,
    "features": features_task,
    "census_data": census_data_task,
    "training_data": training_data_task,
    "summarize_training_data": summarize_training_data_task,
    "plot_training_data": plot_training_data_task,
}
