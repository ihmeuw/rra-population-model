"""Diagnostics output layout under ``{model_version_root}/validation/diagnostics/``.

Checks never build a path themselves: one parquet per check per unit of work
under ``checks/<check_id>/``, plus ``summary.md`` / ``summary.json`` at the root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from rra_population_model.data import PopulationModelData, save_parquet

if TYPE_CHECKING:
    from pathlib import Path


def diagnostics_root(pm_data: PopulationModelData, resolution: str, version: str) -> Path:
    return pm_data.validation_root(resolution, version) / "diagnostics"


def check_dir(pm_data: PopulationModelData, resolution: str, version: str, check_id: str) -> Path:
    return diagnostics_root(pm_data, resolution, version) / "checks" / check_id


def check_path(
    pm_data: PopulationModelData, resolution: str, version: str, check_id: str, *parts: str
) -> Path:
    """``checks/<check_id>/<part1>_<part2>.parquet`` -- e.g. ("DEU", "2022q1"), ("2020q1",) or (block_key,)."""
    return check_dir(pm_data, resolution, version, check_id) / ("_".join(parts) + ".parquet")


def save_check(
    pm_data: PopulationModelData,
    data: pd.DataFrame,
    resolution: str,
    version: str,
    check_id: str,
    *parts: str,
) -> None:
    path = check_path(pm_data, resolution, version, check_id, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(data.reset_index(drop=True), path)


def check_partitions(pm_data: PopulationModelData, resolution: str, version: str, check_id: str) -> list[str]:
    directory = check_dir(pm_data, resolution, version, check_id)
    return sorted(p.stem for p in directory.glob("*.parquet")) if directory.exists() else []


def load_check(pm_data: PopulationModelData, resolution: str, version: str, check_id: str) -> pd.DataFrame:
    """Concatenate every partition of one check; an empty frame when nothing was written."""
    directory = check_dir(pm_data, resolution, version, check_id)
    paths = sorted(directory.glob("*.parquet")) if directory.exists() else []
    frames = [pd.read_parquet(p) for p in paths]
    frames = [f for f in frames if len(f.columns)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summary_path(pm_data: PopulationModelData, resolution: str, version: str, suffix: str) -> Path:
    return diagnostics_root(pm_data, resolution, version) / f"summary.{suffix}"
