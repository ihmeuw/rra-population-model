from rra_population_model.postprocess.compile_results import (
    compile_results,
    compile_results_task,
    upsample_task,
)
from rra_population_model.postprocess.rake import (
    make_raking_factors_task,
    rake,
    rake_task,
)
from rra_population_model.postprocess.rake_itu import (
    rake_itu,
    rake_itu_task,
)

RUNNERS = {
    "rake": rake,
    "rake_itu": rake_itu,
    "compile": compile_results,
}

TASK_RUNNERS = {
    "make_raking_factors": make_raking_factors_task,
    "rake": rake_task,
    "rake_itu": rake_itu_task,
    "compile": compile_results_task,
    "upsample": upsample_task,
}
