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
