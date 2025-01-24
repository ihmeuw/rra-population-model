import click

from rra_population_model import (
    model,
    postprocess,
    preprocess,
)


@click.group()
def pmrun() -> None:
    """Run a stage of the population modeling pipeline."""


@click.group()
def pmtask() -> None:
    """Run an individual modeling task in the population modeling pipeline."""


for module in [preprocess, model, postprocess]:
    runners = getattr(module, "RUNNERS", {})
    task_runners = getattr(module, "TASK_RUNNERS", {})

    if not runners or not task_runners:
        continue

    command_name = module.__name__.split(".")[-1]

    @click.group(name=command_name)
    def workflow_runner() -> None:
        pass

    for name, runner in runners.items():
        workflow_runner.add_command(runner, name)

    pmrun.add_command(workflow_runner)

    @click.group(name=command_name)
    def task_runner() -> None:
        pass

    for name, runner in task_runners.items():
        task_runner.add_command(runner, name)

    pmtask.add_command(task_runner)