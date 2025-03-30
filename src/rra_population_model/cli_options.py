from collections.abc import Callable, Collection

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    convert_choice,
    process_choices,
    with_choice,
    with_debugger,
    with_dry_run,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_overwrite,
    with_progress_bar,
    with_queue,
    with_verbose,
)

from rra_population_model import constants as pmc


def with_resolution[**P, T](
    choices: Collection[str] = pmc.RESOLUTIONS.to_list(),
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "resolution",
        allow_all=allow_all,
        choices=choices,
        help="Resolution of each pixel in the units of the selected CRS.",
        required=True,
    )


def with_iso3[**P, T](
    choices: Collection[str] | None = None,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "iso3",
        allow_all=allow_all,
        choices=choices,
        help="ISO3 code of country to run.",
        callback=lambda ctx, param, value: value.upper(),  # noqa: ARG005
    )


def with_year[**P, T](
    choices: Collection[str] | None = None,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=choices,
        help="Year to run.",
    )


def with_time_point[**P, T](
    choices: Collection[str] | None = pmc.ALL_TIME_POINTS,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "time_point",
        "t",
        allow_all=allow_all,
        choices=choices,
        help="Time point to run.",
        convert=choices is not None and allow_all,
    )


def with_denominator[**P, T](
    choices: Collection[str] = pmc.DENOMINATORS,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "denominator",
        "d",
        allow_all=allow_all,
        choices=choices,
        help="Denominator to run.",
        convert=choices is not None and allow_all,
    )


def with_ntl_option[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "ntl_option",
        "n",
        allow_all=allow_all,
        choices=["none", "ntl", "log_ntl"],
        help="Nighttime light option to run.",
        convert=allow_all,
    )


def with_version[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--version",
        type=click.STRING,
        required=True,
        help="Version of model to run.",
    )


def with_copy_from_version[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--copy-from-version",
        type=click.STRING,
        help="Version of the model to copy predictions from. Used if we're "
        "raking a set of predictions to multiple raking targets.",
    )


def with_block_key[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--block-key",
        "-b",
        type=click.STRING,
        required=True,
        help="Block key to run.",
    )


def with_tile_key[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--tile-key",
        "-k",
        type=click.STRING,
        required=True,
        help="Tile key of tile to run.",
    )


__all__ = [
    "RUN_ALL",
    "convert_choice",
    "process_choices",
    "with_block_key",
    "with_choice",
    "with_debugger",
    "with_dry_run",
    "with_input_directory",
    "with_iso3",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_resolution",
    "with_tile_key",
    "with_time_point",
    "with_verbose",
    "with_year",
]
