import datetime
from pathlib import Path


def get_last_run_version(output_root: str | Path) -> tuple[str, int]:
    """Gets a path to a datetime directory for a new output.

    Parameters
    ----------
    output_root
        The root directory for all outputs.

    """
    output_root = Path(output_root).resolve()
    launch_time = datetime.datetime.now().strftime("%Y_%m_%d")  # noqa: DTZ005
    today_runs = [
        int(run_dir.name.split(".")[-1])
        for run_dir in output_root.iterdir()
        if run_dir.name.startswith(launch_time)
    ]
    if today_runs:
        return launch_time, max(today_runs)
    else:
        return launch_time, 0


def reserve_versions(output_root: str | Path, n: int) -> list[str]:
    """Claim `n` new version directories for today, in order.

    Each is claimed by creating its directory, which the filesystem does
    atomically: if two launches race for the same number, exactly one mkdir
    succeeds and the other moves on to the next number. Picking a number and
    leaving the directory to be made later, by the training task, let three
    launches seconds apart all take the same version and overwrite each other
    (2026-10-01).
    """
    output_root = Path(output_root).resolve()
    today, last = get_last_run_version(output_root)
    versions: list[str] = []
    number = last + 1
    while len(versions) < n:
        version = f"{today}.{number:03d}"
        try:
            (output_root / version).mkdir()
        except FileExistsError:
            pass
        else:
            versions.append(version)
        number += 1
    return versions
