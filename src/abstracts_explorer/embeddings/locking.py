"""
Exclusive locking for local ChromaDB stores.

ChromaDB's local persistent client is not safe for concurrent processes: each
process loads the HNSW index into memory and persists its own copy, so a second
writer can silently corrupt the index. To prevent that, every local client holds
an exclusive lock on its storage directory for the lifetime of the client.

Within a single process, multiple managers share one reentrant lock so threaded
use (for example the web UI) keeps working. Concurrent access from separate
processes is refused; use an HTTP ChromaDB server for that.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

LOCK_FILENAME = ".chroma.lock"


class LocalChromaInUseError(RuntimeError):
    """Raised when another process holds the lock for a local ChromaDB."""


@dataclass
class _LockEntry:
    lock: FileLock
    refs: int = 0


_ENTRIES: Dict[str, _LockEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def _lock_key(path: str) -> str:
    return os.path.realpath(str(path))


def _lock_file(path: str) -> str:
    return os.path.join(str(path), LOCK_FILENAME)


def acquire_local_chroma_lock(path: str) -> None:
    """
    Acquire the exclusive process-lifetime lock for a local ChromaDB *path*.

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
        If another process currently holds the lock.
    """
    key = _lock_key(path)
    with _REGISTRY_LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            entry = _LockEntry(lock=FileLock(_lock_file(path), thread_local=False))
            _ENTRIES[key] = entry

        if entry.refs == 0:
            try:
                entry.lock.acquire(timeout=0)
            except Timeout as exc:
                _ENTRIES.pop(key, None)
                raise LocalChromaInUseError(
                    f"Another process is using the local ChromaDB at {path}. "
                    "Stop it before starting a new one, or set EMBEDDING_DB to an "
                    "HTTP ChromaDB server for concurrent access."
                ) from exc

        entry.refs += 1


def release_local_chroma_lock(path: str) -> None:
    """
    Release one reference to the local ChromaDB lock for *path*.

    The underlying lock is released when the last reference is dropped. Calls
    for an unheld path are ignored.

    Parameters
    ----------
    path : str
        Local ChromaDB storage directory.
    """
    key = _lock_key(path)
    with _REGISTRY_LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            return

        entry.refs -= 1
        if entry.refs > 0:
            return

        try:
            entry.lock.release()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to release local ChromaDB lock for %s", path, exc_info=True)
        finally:
            _ENTRIES.pop(key, None)
