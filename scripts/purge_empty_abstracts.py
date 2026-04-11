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
from abstracts_explorer.embeddings import EmbeddingsError, EmbeddingsManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Delete papers without abstracts from the paper database and the "
            "ChromaDB embeddings collection."
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
    if not db._session:
        raise DatabaseError("Not connected to database")

    try:
        stmt = select(Paper.uid).where(
            or_(
                Paper.abstract.is_(None),
                Paper.abstract == "",
            )
        )
        result = db._session.execute(stmt).scalars().all()
        return list(result)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to query papers without abstracts: {e}") from e


def delete_papers_by_uids(db: DatabaseManager, uids: list[str]) -> int:
    """Delete papers from the paper database by their UIDs.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.
    uids : list[str]
        UIDs of the papers to delete.

    Returns
    -------
    int
        Number of papers deleted.

    Raises
    ------
    DatabaseError
        If deletion fails.
    """
    if not db._session:
        raise DatabaseError("Not connected to database")

    if not uids:
        return 0

    try:
        from sqlalchemy import delete as sa_delete

        result = db._session.execute(sa_delete(Paper).where(Paper.uid.in_(uids)))
        db._session.commit()
        count = result.rowcount if result.rowcount is not None else 0
        return count
    except DatabaseError:
        raise
    except Exception as e:
        db._session.rollback()
        raise DatabaseError(f"Failed to delete papers: {e}") from e


def delete_embeddings_by_uids(em: EmbeddingsManager, uids: list[str]) -> int:
    """Delete embeddings from ChromaDB by their UIDs.

    Parameters
    ----------
    em : EmbeddingsManager
        Connected embeddings manager with an active collection.
    uids : list[str]
        UIDs of the embeddings to delete.

    Returns
    -------
    int
        Number of embeddings deleted.

    Raises
    ------
    EmbeddingsError
        If deletion fails.
    """
    if not uids:
        return 0

    try:
        # Only delete IDs that actually exist in the collection
        existing = em.collection.get(ids=uids)
        ids_present = existing.get("ids", [])
        if ids_present:
            em.collection.delete(ids=ids_present)
        return len(ids_present)
    except EmbeddingsError:
        raise
    except Exception as e:
        raise EmbeddingsError(f"Failed to delete embeddings: {e}") from e


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
        em.connect()
        em.create_collection(reset=False)
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
