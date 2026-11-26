"""
Server-start warm-up for the web UI.

The first request against a cold web UI process otherwise pays the cost of
opening the database, verifying the schema, and running the metadata queries
used by the initial page.  This module performs that work once at startup so
the first page load is fast.

Heavy feature dependencies (clustering, search keyword extraction, plugins,
MCP topic evolution) are intentionally **not** imported here: they are not
needed for the initial page and would only delay server startup.
"""

import logging
import time

logger = logging.getLogger(__name__)


def warm_up() -> None:
    """
    Prime database connections, schema verification, and metadata queries.

    Runs once before the web server starts serving requests.  Failures are
    logged and swallowed so a warm-up problem never prevents the server from
    starting; individual requests will then initialize lazily as before.

    The function creates a :class:`~abstracts_explorer.database.DatabaseManager`,
    opens a connection, verifies/creates the schema (which also populates the
    process-level schema cache), and runs the conference/year/location metadata
    queries used by the initial page and filter endpoints.

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

    logger.info("Web UI warm-up completed in %.2fs", time.perf_counter() - started)
