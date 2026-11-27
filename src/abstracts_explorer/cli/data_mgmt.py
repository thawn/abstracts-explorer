"""
``data delete`` command

"""

import argparse
import logging
import sys
from contextlib import closing
from typing import List


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager


from abstracts_explorer.cli._common import (
    _confirm,
    _print_header,
    _resolve_conference_arg,
)

logger = logging.getLogger(__name__)


def delete_data_command(args: argparse.Namespace) -> int:
    """
    Delete all data for a specific conference and year from every database.

    Removes papers from the paper database, embeddings from ChromaDB, and
    clustering cache entries from the database.  Both ``--conference`` and
    ``--year`` are required.  The user must type ``yes`` to confirm.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - conference: Conference name (required)
        - year: Year of conference (required)
        - yes: If True, skip the interactive confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    conference = getattr(args, "conference", None)
    year = getattr(args, "year", None)

    if not conference:
        print("❌ --conference is required for the data delete command.", file=sys.stderr)
        return 1
    if year is None:
        print("❌ --year is required for the data delete command.", file=sys.stderr)
        return 1

    assert conference is not None
    assert year is not None

    # Resolve conference name to canonical form
    try:
        conference = _resolve_conference_arg(conference)
    except Exception:
        # If resolution fails, use the raw value (might still be valid)
        pass

    assert conference is not None

    _print_header(
        "Delete Conference/Year Data",
        f"Database: {config.masked_database_url}",
        f"Embedding DB: {config.embedding_db}",
        f"Conference: {conference}",
        f"Year: {year}",
    )

    # Count what will be deleted
    paper_count = 0
    presentation_count = 0
    cache_count = 0
    paper_uids: List[str] = []
    try:
        with DatabaseManager() as db:
            db.create_tables()
            filters = {"conference": conference, "year": year}
            paper_uids = db.get_paper_uids(filters)
            paper_count = db.get_paper_count(filters)
            presentation_count = db.get_total_presentation_count(filters)
            cache_count = db.count_clustering_cache_by_conference_year(conference, year)
    except Exception as e:
        print(f"\n❌ Error accessing database: {e}", file=sys.stderr)
        return 1

    # Count ChromaDB embeddings for the resolved UIDs
    embedding_count = 0
    try:
        with closing(EmbeddingsManager()) as em:
            existing = em.collection.get(ids=paper_uids, include=[])
            embedding_count = len(existing.get("ids", []))
    except Exception as e:
        logger.warning(f"Could not count ChromaDB embeddings (will still attempt deletion): {e}")
        embedding_count = 0

    print("\nThe following data will be permanently deleted:")
    print(f"  📄 Papers:            {paper_count:,}")
    if presentation_count != paper_count:
        print(f"  🎤 Presentations:     {presentation_count:,}")
    print(f"  🔢 Embeddings:        {embedding_count:,}")
    print(f"  🗂️  Clustering cache:  {cache_count:,}")

    if paper_count == 0 and embedding_count == 0 and cache_count == 0:
        print("\n✅ Nothing to delete — no data found for this conference/year combination.")
        return 0

    if not getattr(args, "yes", False):
        print(f"\n⚠️  This will permanently delete all data for {conference} {year} from all databases.")
        if not _confirm("Type 'yes' to confirm: ", require_yes=True):
            print("Aborted.")
            return 1

    errors: List[str] = []

    # 1. Delete embeddings from ChromaDB using the UIDs resolved above
    try:
        with closing(EmbeddingsManager()) as em:
            deleted_embeddings = em.delete_embeddings_by_ids(paper_uids)
        print(f"\n✅ Deleted {deleted_embeddings:,} embedding(s) from ChromaDB.")
    except Exception as e:
        msg = f"Failed to delete embeddings: {e}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    # 2. Delete papers from paper database
    try:
        with DatabaseManager() as db:
            deleted_papers = db.delete_papers_by_conference_year(conference, year)
        print(f"✅ Deleted {deleted_papers:,} paper(s) from paper database.")
    except Exception as e:
        msg = f"Failed to delete papers: {e}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    # 3. Delete clustering cache
    try:
        with DatabaseManager() as db:
            deleted_cache = db.delete_clustering_cache_by_conference_year(conference, year)
        print(f"✅ Deleted {deleted_cache:,} clustering cache entry/entries.")
    except Exception as e:
        msg = f"Failed to delete clustering cache: {e}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    if errors:
        print(f"\n⚠️  Completed with {len(errors)} error(s).")
        return 1

    print(f"\n✅ All data for {conference} {year} has been deleted.")
    return 0
