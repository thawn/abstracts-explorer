#!/usr/bin/env python3
"""Delete papers without abstracts from paper and embedding databases.

Papers that have empty or missing abstracts match unusually well against
almost arbitrary short queries, producing misleading search results.  This
script finds all such papers in the SQLite/PostgreSQL paper database and
removes them along with their ChromaDB embeddings.

Usage::

    python scripts/purge_empty_abstracts.py
    python scripts/purge_empty_abstracts.py --paper-db /path/to/abstracts.db
    python scripts/purge_empty_abstracts.py --yes          # skip confirmation
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import or_, select

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import Paper
from abstracts_explorer.embeddings import EmbeddingsManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Delete papers without abstracts from the paper database and the ChromaDB embeddings collection."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args()


def find_papers_without_abstract(db: DatabaseManager) -> list[str]:
    """Return the UIDs of all papers that have an empty or missing abstract.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.

    Returns
    -------
    list[str]
        UIDs of papers with empty or ``NULL`` abstracts.

    Raises
    ------
    DatabaseError
        If the query fails.
    """
    try:
        stmt = select(Paper.uid).where(
            or_(
                Paper.abstract.is_(None),
                Paper.abstract == "",
            )
        )
        result = db.session.execute(stmt).scalars().all()
        return list(result)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to query papers without abstracts: {e}") from e


def delete_papers_by_uids(db: DatabaseManager, uids: list[str]) -> int:
    """Delete papers by UID using the production database helper."""
    return db.delete_papers_by_uids(uids)


def delete_embeddings_by_uids(em: EmbeddingsManager, uids: list[str]) -> int:
    """Delete existing embeddings by UID using the production bulk helper."""
    if not uids:
        return 0

    existing = em.collection.get(ids=uids)
    ids_present = existing.get("ids", [])
    if not ids_present:
        return 0
    return em.delete_embeddings_by_ids(list(ids_present))



def main() -> int:
    """Run the purge.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
        from abstracts_explorer.config import get_config

        get_config(reload=True)

    # ------------------------------------------------------------------
    # Step 1: find papers without abstracts
    # ------------------------------------------------------------------
    try:
        with DatabaseManager() as db:
            db.create_tables()
            uids = find_papers_without_abstract(db)
    except DatabaseError as exc:
        print(f"❌ Failed to query paper database: {exc}", file=sys.stderr)
        return 1

    if not uids:
        print("✅ No papers without abstracts found — nothing to do.")
        return 0

    print(f"Found {len(uids):,} paper(s) with empty or missing abstracts.")

    # ------------------------------------------------------------------
    # Step 2: confirm
    # ------------------------------------------------------------------
    if not args.yes:
        print(
            "\n⚠️  These papers will be permanently deleted from both the paper "
            "database and the ChromaDB embeddings collection."
        )
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Step 3: delete from paper database
    # ------------------------------------------------------------------
    try:
        with DatabaseManager() as db:
            deleted_papers = delete_papers_by_uids(db, uids)
        print(f"\n✅ Deleted {deleted_papers:,} paper(s) from the paper database.")
    except DatabaseError as exc:
        msg = f"Failed to delete papers: {exc}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    # ------------------------------------------------------------------
    # Step 4: delete from ChromaDB
    # ------------------------------------------------------------------
    try:
        em = EmbeddingsManager()
        deleted_embeddings = delete_embeddings_by_uids(em, uids)
        em.close()
        print(f"✅ Deleted {deleted_embeddings:,} embedding(s) from ChromaDB.")
    except Exception as exc:
        msg = f"Failed to delete embeddings: {exc}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    if errors:
        print(f"\n⚠️  Finished with {len(errors)} error(s).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
