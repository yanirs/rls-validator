"""Data loading functionality."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


# TODO: add unit tests from Chatty
def get_app_data_dir(app_name: str) -> Path:
    """Get the OS-specific app data directory and create a subdirectory for the app."""
    if sys.platform == "win32":
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform in ["linux", "linux2", "darwin"]:
        base_dir = Path.home() / ".local" / "share"
    else:
        raise OSError("Unsupported Operating System")
    app_data_dir = base_dir / app_name
    app_data_dir.mkdir(parents=True, exist_ok=True)
    return app_data_dir


def download_text_file(
    url: str, app_name: str = "rls-validator", fresh_days: int = 7
) -> Path:
    """
    Download a text file from the URL to a local app data directory and return its path.

    The function checks if the local copy of the file is older than a specified number
    of days. If so, or if the file doesn't exist, it downloads a fresh copy.

    Parameters
    ----------
    url : str
        URL of the text file to be downloaded.
    app_name : str
        Name of the application, used to create a dedicated subdirectory in the
        OS-specific app data directory.
    fresh_days : int
        The number of days after which the local file is considered outdated and needs
        refreshing.

    Returns
    -------
    pathlib.Path
        The path to the downloaded or existing local file.

    Raises
    ------
    Exception
        If the file does not exist locally and the download failed.
    """
    local_path = get_app_data_dir(app_name) / url.split("/")[-1]
    if local_path.exists():
        last_modified = datetime.fromtimestamp(
            local_path.stat().st_mtime, tz=timezone.utc
        )
        if (datetime.now(timezone.utc) - last_modified).days < fresh_days:
            return local_path
    try:
        response = httpx.get(url)
        response.raise_for_status()
        local_path.write_text(response.text)
    except Exception:
        if not local_path.exists():
            raise
    return local_path
