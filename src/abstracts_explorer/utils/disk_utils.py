"""
Disk-space helpers.

Small utilities used to fail fast when a write would risk filling the disk.
Running out of space while a native database (for example ChromaDB's HNSW
index) is being persisted can corrupt the store, so callers should check
available capacity before starting a large write.
"""

import shutil
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


class InsufficientDiskSpaceError(OSError):
    """Raised when the available disk space is below the required minimum."""


def _existing_ancestor(path: PathLike) -> Path:
    """Return *path* or its nearest existing ancestor."""
    probe = Path(path).expanduser()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def available_bytes(path: PathLike) -> int:
    """
    Return the number of free bytes on the filesystem containing *path*.

    Parameters
    ----------
    path : str or pathlib.Path
        An existing file or directory, or a path that will be created. When
        the path does not exist yet, the nearest existing ancestor is probed.

    Returns
    -------
    int
        Free bytes available to the current user on that filesystem.
    """
    return shutil.disk_usage(_existing_ancestor(path)).free


def ensure_free_space(path: PathLike, min_free_bytes: int, *, description: str = "write") -> int:
    """
    Ensure at least *min_free_bytes* are available on the filesystem for *path*.

    Parameters
    ----------
    path : str or pathlib.Path
        Path that will be written to.
    min_free_bytes : int
        Required amount of free space in bytes. Non-positive values disable
        the check.
    description : str, optional
        Human-readable description of the operation, included in the error
        message (default ``"write"``).

    Returns
    -------
    int
        The available free space in bytes.

    Raises
    ------
    InsufficientDiskSpaceError
        If fewer than *min_free_bytes* are available.
    """
    free = available_bytes(path)
    if min_free_bytes <= 0:
        return free

    if free < min_free_bytes:
        raise InsufficientDiskSpaceError(
            f"Not enough disk space for {description}: {free} bytes available at "
            f"{_existing_ancestor(path)}, but at least {min_free_bytes} bytes are required. "
            "Free up space or set MIN_FREE_DISK_MB=0 to disable this check."
        )
    return free
