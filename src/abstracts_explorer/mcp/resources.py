"""
MCP resource loading and lifecycle helpers.
"""

import atexit
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from abstracts_explorer.mcp import shared_dependencies
from abstracts_explorer.mcp.errors import ClusterAnalysisError

logger = logging.getLogger(__name__)


@dataclass
class _OwnedResource:
    """A cached MCP resource and the thread that created it."""

    owner: threading.Thread
    resource: Any


_RESOURCE_STATE = threading.local()
_RESOURCE_LOCK = threading.Lock()
_ALL_RESOURCES: list[_OwnedResource] = []
_RESOURCE_GENERATION = 0


def _safe_close(resources: list[Any]) -> None:
    """Close resources, logging but suppressing cleanup errors."""
    for resource in resources:
        try:
            resource.close()
        except Exception as exc:
            logger.debug("Failed to close MCP resource: %s", exc)


def _close_current_thread_state() -> None:
    """Close and forget the current thread's cached resources."""
    state = getattr(_RESOURCE_STATE, "state", None)
    if state is None:
        return

    resources: list[Any] = []
    db = state.get("db")
    if db is not None:
        resources.append(db)
    resources.extend(state.get("embeddings", {}).values())

    state["db"] = None
    state["embeddings"] = {}

    current = threading.current_thread()
    with _RESOURCE_LOCK:
        _ALL_RESOURCES[:] = [entry for entry in _ALL_RESOURCES if entry.owner is not current]

    _safe_close(resources)


def _thread_state() -> Dict[str, Any]:
    """Return the current thread's cached resource state."""
    state = getattr(_RESOURCE_STATE, "state", None)
    if state is not None and state.get("generation") == _RESOURCE_GENERATION:
        return state

    _close_current_thread_state()
    state = {"db": None, "embeddings": {}, "generation": _RESOURCE_GENERATION}
    _RESOURCE_STATE.state = state
    return state


def _register_resource(resource: Any) -> None:
    """Track a resource for later cleanup."""
    with _RESOURCE_LOCK:
        _ALL_RESOURCES.append(_OwnedResource(threading.current_thread(), resource))


def _get_shared_database() -> "shared_dependencies.DatabaseManager":
    """Return a process-cached database manager for the current thread."""
    state = _thread_state()
    db = state["db"]
    if db is None:
        db = shared_dependencies.DatabaseManager()
        state["db"] = db
        _register_resource(db)
        logger.debug("Created thread-local MCP database manager")
    return db


def _get_shared_embeddings(collection_name: str) -> "shared_dependencies.EmbeddingsManager":
    """Return a thread/collection-cached embeddings manager."""
    state = _thread_state()
    embeddings = state["embeddings"]
    em = embeddings.get(collection_name)
    if em is None:
        em = shared_dependencies.EmbeddingsManager(collection_name=collection_name)
        embeddings[collection_name] = em
        _register_resource(em)
        logger.debug("Created thread-local MCP embeddings manager for %s", collection_name)
    return em


def close_mcp_resources(force: bool = False) -> None:
    """
    Close and forget cached MCP resources.

    By default only resources owned by the calling thread, or left behind by
    threads that have already exited, are closed.  Resources owned by other
    live threads are left untouched; those threads close their stale resources
    themselves the next time they access the resource cache.

    Parameters
    ----------
    force : bool, optional
        Close resources owned by every thread.  Only use this after all worker
        threads have stopped, for example at application or test teardown.
    """
    global _RESOURCE_GENERATION

    current = threading.current_thread()
    with _RESOURCE_LOCK:
        _RESOURCE_GENERATION += 1
        generation = _RESOURCE_GENERATION
        if force:
            selected = list(_ALL_RESOURCES)
            _ALL_RESOURCES.clear()
        else:
            selected = [entry for entry in _ALL_RESOURCES if entry.owner is current or not entry.owner.is_alive()]
            selected_ids = {id(entry) for entry in selected}
            _ALL_RESOURCES[:] = [entry for entry in _ALL_RESOURCES if id(entry) not in selected_ids]

    state = getattr(_RESOURCE_STATE, "state", None)
    if state is not None:
        state["db"] = None
        state["embeddings"] = {}
        state["generation"] = generation

    _safe_close([entry.resource for entry in selected])


atexit.register(close_mcp_resources)


def load_clustering_data(
    collection_name: Optional[str] = None,
) -> tuple[shared_dependencies.ClusteringManager, shared_dependencies.DatabaseManager]:
    """
    Load clustering data and database.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection

    Returns
    -------
    tuple[shared_dependencies.ClusteringManager, shared_dependencies.DatabaseManager]
        Clustering manager and database manager instances.  The database and
        embeddings managers are process-cached per thread; the clustering
        manager is constructed fresh for each call.

    Raises
    ------
    ClusterAnalysisError
        If loading fails
    """
    config = shared_dependencies.get_config()

    # Use config defaults if not provided
    collection_name = collection_name or config.collection_name

    try:
        em = _get_shared_embeddings(collection_name)
        db = _get_shared_database()
        cm = shared_dependencies.ClusteringManager(em, db)
        return cm, db

    except Exception as e:
        raise ClusterAnalysisError(f"Failed to load clustering data: {str(e)}") from e


@contextmanager
def _get_mcp_resources(
    collection_name: Optional[str] = None,
    *,
    with_embeddings: bool = True,
) -> Iterator[Tuple[Optional[shared_dependencies.EmbeddingsManager], shared_dependencies.DatabaseManager]]:
    """
    Provide cached MCP tool resources.

    The database and embeddings managers are reused within the current thread
    until :func:`close_mcp_resources` is called.

    Parameters
    ----------
    collection_name : str, optional
        ChromaDB collection name.  Uses the configured default when embeddings
        are requested.
    with_embeddings : bool, optional
        Whether to create an ``shared_dependencies.EmbeddingsManager`` (default: ``True``).

    Yields
    ------
    tuple
        ``(embeddings_manager, database_manager)``; the embeddings manager is
        ``None`` when *with_embeddings* is ``False``.
    """
    em: Optional[shared_dependencies.EmbeddingsManager] = None
    if with_embeddings:
        resolved_collection = collection_name or shared_dependencies.get_config().collection_name
        em = _get_shared_embeddings(resolved_collection)
    db = _get_shared_database()
    yield em, db


@contextmanager
def _get_clustering_resources(
    collection_name: Optional[str] = None,
) -> Iterator[Tuple[shared_dependencies.ClusteringManager, shared_dependencies.DatabaseManager]]:
    """
    Provide clustering resources backed by the cached MCP database/embeddings.

    Parameters
    ----------
    collection_name : str, optional
        ChromaDB collection name.

    Yields
    ------
    tuple
        ``(clustering_manager, database_manager)``.
    """
    cm, db = load_clustering_data(collection_name)
    yield cm, db
