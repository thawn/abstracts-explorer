"""
MCP resource loading and lifecycle helpers.
"""

import atexit
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from abstracts_explorer.mcp import shared_dependencies
from abstracts_explorer.mcp.errors import ClusterAnalysisError

logger = logging.getLogger(__name__)

_RESOURCE_STATE = threading.local()
_RESOURCE_LOCK = threading.Lock()
_ALL_RESOURCES: list[Any] = []
_RESOURCE_GENERATION = 0


def _thread_state() -> Dict[str, Any]:
    """Return the current thread's cached resource state."""
    state = getattr(_RESOURCE_STATE, "state", None)
    if state is None or state.get("generation") != _RESOURCE_GENERATION:
        state = {"db": None, "embeddings": {}, "generation": _RESOURCE_GENERATION}
        _RESOURCE_STATE.state = state
    return state


def _get_shared_database() -> "shared_dependencies.DatabaseManager":
    """Return a process-cached database manager for the current thread."""
    state = _thread_state()
    db = state["db"]
    if db is None:
        db = shared_dependencies.DatabaseManager()
        state["db"] = db
        with _RESOURCE_LOCK:
            _ALL_RESOURCES.append(db)
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
        with _RESOURCE_LOCK:
            _ALL_RESOURCES.append(em)
        logger.debug("Created thread-local MCP embeddings manager for %s", collection_name)
    return em


def close_mcp_resources() -> None:
    """
    Close and forget all cached MCP database and embeddings resources.

    Resources are cached per thread because ``DatabaseManager`` owns a single
    SQLAlchemy session that is not safe to share across threads.  Every live
    thread lazily recreates its resources after this function increments the
    resource generation, so calling it from one thread also invalidates caches
    held by other threads.
    """
    global _RESOURCE_GENERATION
    with _RESOURCE_LOCK:
        _RESOURCE_GENERATION += 1
        resources = list(reversed(_ALL_RESOURCES))
        _ALL_RESOURCES.clear()

    state = getattr(_RESOURCE_STATE, "state", None)
    if state is not None:
        state["db"] = None
        state["embeddings"] = {}
        state["generation"] = _RESOURCE_GENERATION

    for resource in resources:
        try:
            resource.close()
        except Exception as exc:
            logger.debug("Failed to close MCP resource: %s", exc)


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
