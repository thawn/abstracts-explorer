"""
MCP resource loading and lifecycle helpers.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

from abstracts_explorer.mcp import deps
from abstracts_explorer.mcp.errors import ClusterAnalysisError

logger = logging.getLogger(__name__)


def load_clustering_data(
    collection_name: Optional[str] = None,
) -> tuple[deps.ClusteringManager, deps.DatabaseManager]:
    """
    Load clustering data and database.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection

    Returns
    -------
    tuple[deps.ClusteringManager, deps.DatabaseManager]
        Clustering manager and database manager instances

    Raises
    ------
    ClusterAnalysisError
        If loading fails
    """
    config = deps.get_config()

    # Use config defaults if not provided
    collection_name = collection_name or config.collection_name

    em: Optional[deps.EmbeddingsManager] = None
    db: Optional[deps.DatabaseManager] = None

    try:
        # Initialize embeddings manager
        em = deps.EmbeddingsManager(
            collection_name=collection_name,
        )

        # Initialize database manager
        db = deps.DatabaseManager()

        # Initialize clustering manager
        cm = deps.ClusteringManager(em, db)

        return cm, db

    except Exception as e:
        if em is not None:
            em.close()
        if db is not None:
            db.close()
        raise ClusterAnalysisError(f"Failed to load clustering data: {str(e)}") from e


@contextmanager
def _get_mcp_resources(
    collection_name: Optional[str] = None,
    *,
    with_embeddings: bool = True,
) -> Iterator[Tuple[Optional[deps.EmbeddingsManager], deps.DatabaseManager]]:
    """
    Provide MCP tool resources and guarantee they are closed.

    Parameters
    ----------
    collection_name : str, optional
        ChromaDB collection name.  Uses the configured default when embeddings
        are requested.
    with_embeddings : bool, optional
        Whether to create an ``deps.EmbeddingsManager`` (default: ``True``).

    Yields
    ------
    tuple
        ``(embeddings_manager, database_manager)``; the embeddings manager is
        ``None`` when *with_embeddings* is ``False``.
    """
    em: Optional[deps.EmbeddingsManager] = None
    db: Optional[deps.DatabaseManager] = None
    try:
        if with_embeddings:
            resolved_collection = collection_name or deps.get_config().collection_name
            em = deps.EmbeddingsManager(collection_name=resolved_collection)
        db = deps.DatabaseManager()
        yield em, db
    finally:
        if em is not None:
            em.close()
        if db is not None:
            db.close()


@contextmanager
def _get_clustering_resources(
    collection_name: Optional[str] = None,
) -> Iterator[Tuple[deps.ClusteringManager, deps.DatabaseManager]]:
    """
    Provide clustering resources and guarantee they are closed.

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
    try:
        yield cm, db
    finally:
        cm.embeddings_manager.close()
        db.close()
