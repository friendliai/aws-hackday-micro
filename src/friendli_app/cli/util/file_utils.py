import os
from typing import Any


def get_directory_size(path: str, exclude_dirs: list[str] | None = None) -> int:
    """Calculates the total size of a directory in bytes."""
    total_size = 0
    if exclude_dirs is None:
        exclude_dirs = []

    for dirpath, dirnames, filenames in os.walk(path):
        # Exclude directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def scan_directory_structure(path: str, exclude_dirs: list[str] | None = None) -> dict[str, Any]:
    """
    Scans a directory and returns its structure as a dictionary.
    Files are represented as file size (int), and directories as nested dictionaries.
    """
    structure: dict[str, Any] = {}
    if exclude_dirs is None:
        exclude_dirs = []

    for item in os.listdir(path):
        if item in exclude_dirs:
            continue

        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            structure[item] = os.path.getsize(item_path)
        elif os.path.isdir(item_path):
            structure[item] = scan_directory_structure(item_path, exclude_dirs=exclude_dirs)
    return structure
