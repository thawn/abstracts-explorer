"""
MCP resource loading and lifecycle helpers.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

from abstracts_explorer.mcp import shared_dependencies
from abstracts_explorer.mcp.errors import ClusterAnalysisError

logger = logging.getLogger(__name__)


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
        Clustering manager and database manager instances

    Raises
    ------
    ClusterAnalysisError
        If loading fails
    """
    config = shared_dependencies.get_config()

    # Use config defaults if not provided
    collection_name = collection_name or config.collection_name

    em: Optional[shared_dependencies.EmbeddingsManager] = None
    db: Optional[shared_dependencies.DatabaseManager] = None

    try:
        # Initialize embeddings manager
        em = shared_dependencies.EmbeddingsManager(
            collection_name=collection_name,
        )

        # Initialize database manager
        db = shared_dependencies.DatabaseManager()

        # Initialize clustering manager
        cm = shared_dependencies.ClusteringManager(em, db)

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
) -> Iterator[Tuple[Optional[shared_dependencies.EmbeddingsManager], shared_dependencies.DatabaseManager]]:
    """
    Provide MCP tool resources and guarantee they are closed.

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
    db: Optional[shared_dependencies.DatabaseManager] = None
    try:
        if with_embeddings:
            resolved_collection = collection_name or shared_dependencies.get_config().collection_name
            em = shared_dependencies.EmbeddingsManager(collection_name=resolved_collection)
        db = shared_dependencies.DatabaseManager()
        yield em, db
    finally:
        if em is not None:
            em.close()
        if db is not None:
            db.close()


@contextmanager
def _get_clustering_resources(
    collection_name: Optional[str] = None,
) -> Iterator[Tuple[shared_dependencies.ClusteringManager, shared_dependencies.DatabaseManager]]:
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
