from rra_population_model.postprocess.admin_rasters.runner import (
    admin_rasters,
    admin_rasters_task,
)
from rra_population_model.postprocess.mosaic.runner import (
    mosaic,
    mosaic_task,
)
from rra_population_model.postprocess.census_rake.runner import (
    census_rake,
    census_rake_task,
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
from rra_population_model.postprocess.upsample.runner import (
    upsample,
    upsample_task,
)

RUNNERS = {
    "raking_factors": raking_factors,
    "rake": rake,
    "census_rake": census_rake,
    "rake_itu": rake_itu,
    "mosaic": mosaic,
    "upsample": upsample,
    "admin_rasters": admin_rasters,
}

TASK_RUNNERS = {
    "raking_factors": raking_factors_task,
    "rake": rake_task,
    "census_rake": census_rake_task,
    "rake_itu": rake_itu_task,
    "mosaic": mosaic_task,
    "upsample": upsample_task,
    "admin_rasters": admin_rasters_task,
}
