"""
Read/write locking for local ChromaDB stores.

ChromaDB's local persistent client is not safe for concurrent writing: each
process loads the HNSW index into memory and persists its own copy, so a second
writer can silently corrupt the index. To prevent that, writers hold an
exclusive lock on the storage directory for the lifetime of the client and
read-only clients share a read lock. Multiple read-only processes can therefore
run together, but they cannot overlap with a writer.

Within a single process, multiple managers share one lock so threaded use (for
example the web UI) keeps working. Concurrent access from separate processes is
refused when incompatible; use an HTTP ChromaDB server for cross-process
writers.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict

from filelock import ReadWriteLock, Timeout

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".chroma.lock"


class LocalChromaInUseError(RuntimeError):
    """Raised when another process holds an incompatible lock for a local ChromaDB."""


@dataclass
class _LockEntry:
    lock: ReadWriteLock
    readers: int = 0
    writers: int = 0


_ENTRIES: Dict[str, _LockEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def _lock_key(path: str) -> str:
    return os.path.realpath(str(path))


def _lock_file(path: str) -> str:
    return os.path.join(str(path), LOCK_FILENAME)


def _acquire(path: str, mode: str) -> None:
    key = _lock_key(path)
    with _REGISTRY_LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            entry = _LockEntry(lock=ReadWriteLock(_lock_file(path), timeout=0, blocking=False, is_singleton=True))
            _ENTRIES[key] = entry

        if mode == "read":
            if entry.writers > 0:
                raise LocalChromaInUseError(
                    f"The local ChromaDB at {path} is already open for writing in this process. "
                    "Read-only and write access cannot be mixed within one process."
                )
            if entry.readers == 0:
                try:
                    entry.lock.acquire_read(timeout=0)
                except (Timeout, RuntimeError) as exc:
                    _ENTRIES.pop(key, None)
                    raise LocalChromaInUseError(
                        f"Another process is using the local ChromaDB at {path}. "
                        "Stop it before starting a new one, or set EMBEDDING_DB to an "
                        "HTTP ChromaDB server for concurrent access."
                    ) from exc
            entry.readers += 1
            return

        if entry.readers > 0:
            raise LocalChromaInUseError(
                f"The local ChromaDB at {path} is already open in read-only mode in this process. "
                "Read-only and write access cannot be mixed within one process."
            )
        if entry.writers == 0:
            try:
                entry.lock.acquire_write(timeout=0)
            except (Timeout, RuntimeError) as exc:
                _ENTRIES.pop(key, None)
                raise LocalChromaInUseError(
                    f"Another process is using the local ChromaDB at {path}. "
                    "Stop it before starting a new one, or set EMBEDDING_DB to an "
                    "HTTP ChromaDB server for concurrent access."
                ) from exc
        entry.writers += 1


def acquire_local_chroma_lock(path: str) -> None:
    """
    Acquire the exclusive process-lifetime write lock for a local ChromaDB.

    The call is reference-counted so several managers in the same process can
    share the lock. The lock is released by :func:`release_local_chroma_lock`
    (or automatically by the OS when the process exits).

    Parameters
    ----------
    path : str
        Local ChromaDB storage directory. It must already exist.

    Raises
    ------
    LocalChromaInUseError
        If another process currently holds a read or write lock, or the same
        process already holds a read lock.
    """
    _acquire(path, "write")


def acquire_local_chroma_read_lock(path: str) -> None:
    """
    Acquire a shared read lock for a local ChromaDB.

    Multiple read-only processes may hold the lock simultaneously, but the
    lock cannot be combined with a concurrent writer.

    Parameters
    ----------
    path : str
        Local ChromaDB storage directory. It must already exist.

    Raises
    ------
    LocalChromaInUseError
        If a writer currently holds the lock, or the same process already
        holds the write lock.
    """
    _acquire(path, "read")


def _release(path: str, mode: str) -> None:
    key = _lock_key(path)
    with _REGISTRY_LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            return

        if mode == "read":
            if entry.readers <= 0:
                return
            entry.readers -= 1
            if entry.readers > 0:
                return
        else:
            if entry.writers <= 0:
                return
            entry.writers -= 1
            if entry.writers > 0:
                return

        try:
            entry.lock.release()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to release local ChromaDB lock for %s", path, exc_info=True)
        finally:
            _ENTRIES.pop(key, None)


def release_local_chroma_lock(path: str) -> None:
    """
    Release one writer reference to the local ChromaDB lock for *path*.

    The underlying lock is released when the last reference is dropped. Calls
    for an unheld path are ignored.

    Parameters
    ----------
    path : str
        Local ChromaDB storage directory.
    """
    _release(path, "write")


def release_local_chroma_read_lock(path: str) -> None:
    """
    Release one reader reference to the local ChromaDB lock for *path*.

    The underlying lock is released when the last reader reference is dropped.
    Calls for an unheld path are ignored.

    Parameters
    ----------
    path : str
        Local ChromaDB storage directory.
    """
    _release(path, "read")
