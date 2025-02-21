from rra_population_model.postprocess.compile_results import (
    compile_results,
    compile_results_task,
    upsample_task,
)
from rra_population_model.postprocess.rake.runner import (
    rake,
    rake_task,
)
from rra_population_model.postprocess.rake_itu import (
    rake_itu,
    rake_itu_task,
)
from rra_population_model.postprocess.raking_factors.runner import (
    raking_factors,
    raking_factors_task,
)

RUNNERS = {
    "raking_factors": raking_factors,
    "rake": rake,
    "rake_itu": rake_itu,
    "compile": compile_results,
}

TASK_RUNNERS = {
    "raking_factors": raking_factors_task,
    "rake": rake_task,
    "rake_itu": rake_itu_task,
    "compile": compile_results_task,
    "upsample": upsample_task,
}
