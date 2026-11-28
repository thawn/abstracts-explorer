"""
Server-start warm-up for the web UI.

The first request against a cold web UI process otherwise pays the cost of
opening the database, verifying the schema, and running the metadata queries
used by the initial page.  This module performs that work once at startup so
the first page load is fast.

Heavy feature dependencies (clustering, search keyword extraction, plugins,
MCP topic evolution) are intentionally **not** imported here: they are not
needed for the initial page and would only delay server startup.  The optional
ChromaDB warm-up imports the embeddings manager only when enabled.
"""

import logging
import time

logger = logging.getLogger(__name__)


def _warm_up_embeddings(config: object) -> None:
    """
    Warm the configured ChromaDB collection without calling the embedding API.

    The collection is opened, its size is read with ``count()``, and one tiny
    query with a zero vector loads the HNSW index.  The vector dimension is
    obtained from ``peek()`` or collection metadata, so no embedding request is
    made.  Empty collections and all failures are skipped safely.

    Parameters
    ----------
    config : object
        Application configuration providing ``collection_name``.

    Returns
    -------
    None
    """
    started = time.perf_counter()
    manager = None
    try:
        from abstracts_explorer.embeddings import EmbeddingsManager

        manager = EmbeddingsManager(collection_name=getattr(config, "collection_name", None))
        collection = manager.collection
        count = int(collection.count())
        if count <= 0:
            logger.debug("Skipping embedding warm-up for empty collection")
            return

        dimension = None
        peeked = collection.peek(limit=1)
        embeddings = peeked.get("embeddings") if isinstance(peeked, dict) else None
        if embeddings:
            dimension = len(embeddings[0])
        if dimension is None:
            metadata = getattr(collection, "metadata", None) or {}
            dimension = metadata.get("embedding_dimension") or metadata.get("dimension")
        if not dimension:
            logger.debug("Could not determine embedding dimension; skipping Chroma query warm-up")
            return

        collection.query(query_embeddings=[[0.0] * int(dimension)], n_results=1)
        logger.info("Embedding warm-up completed in %.2fs", time.perf_counter() - started)
    except Exception:
        logger.warning("Embedding warm-up failed; continuing with lazy initialization", exc_info=True)
    finally:
        if manager is not None:
            try:
                manager.close()
            except Exception:
                logger.debug("Failed to close embeddings manager after warm-up", exc_info=True)


def warm_up() -> None:
    """
    Prime database connections, schema verification, metadata queries, and ChromaDB.

    Runs once before the web server starts serving requests.  Failures are
    logged and swallowed so a warm-up problem never prevents the server from
    starting; individual requests will then initialize lazily as before.

    The function creates a :class:`~abstracts_explorer.database.DatabaseManager`,
    opens a connection, verifies/creates the schema (which also populates the
    process-level schema cache), and runs the conference/year/location metadata
    queries used by the initial page and filter endpoints.  When
    ``WARMUP_EMBEDDINGS`` is enabled (the default), it also loads the configured
    ChromaDB collection and HNSW index.

    Returns
    -------
    None

    Examples
    --------
    >>> from abstracts_explorer.web_ui.warmup import warm_up
    >>> warm_up()
    """
    started = time.perf_counter()
    try:
        from abstracts_explorer.config import get_config
        from abstracts_explorer.database import DatabaseManager

        config = get_config()
        database = DatabaseManager()
        try:
            database.connect()
            database.create_tables()
            database.get_conference_years_from_db()
            database.get_conference_year_locations()
            database.get_conferences()
            database.confirm_default_conference_year_exists(
                config.default_conference or "",
                config.default_year or None,
            )
        finally:
            database.close()
    except Exception:
        logger.warning("Web UI warm-up failed; continuing with lazy initialization", exc_info=True)
        return

    if getattr(config, "warmup_embeddings", True):
        _warm_up_embeddings(config)

    logger.info("Web UI warm-up completed in %.2fs", time.perf_counter() - started)
