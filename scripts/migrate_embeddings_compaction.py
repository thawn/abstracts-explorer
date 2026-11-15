#!/usr/bin/env python3
"""Compact an existing ChromaDB collection into the hash-only format.

This one-time migration upgrades collections created before the hash-only
storage change:

1. Remove all per-paper metadata from the ChromaDB collection.
2. Replace stored full-text documents with SHA-256 content hashes.
3. Optionally delete embeddings whose paper rows no longer exist in the SQL
   database (``--prune``).
4. Optionally run ``chroma vacuum`` afterwards to reclaim disk space
   (``--vacuum``).

The migration is idempotent: records that are already compacted are left
untouched.

Usage::

    python scripts/migrate_embeddings_compaction.py
    python scripts/migrate_embeddings_compaction.py --prune --vacuum
    python scripts/migrate_embeddings_compaction.py --paper-db /path/to/abstracts.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import batched
from pathlib import Path
from typing import Optional, Sequence

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager, hash_text, is_content_hash

_BATCH_SIZE = 1000


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Remove per-paper metadata and replace full-text documents with content hashes "
            "in an existing ChromaDB collection."
        )
    )
    parser.add_argument(
        "--collection",
        help="ChromaDB collection name (default: from config).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete embeddings whose paper rows no longer exist in the database.",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run 'chroma vacuum' after compaction to reclaim disk space.",
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    return parser.parse_args(argv)


def _all_ids(embeddings_manager: EmbeddingsManager) -> list[str]:
    """Return all record IDs in the collection."""
    return list(embeddings_manager.collection.get(include=[])["ids"])


def remove_all_metadata(embeddings_manager: EmbeddingsManager, batch_size: int = _BATCH_SIZE) -> int:
    """
    Remove all per-paper metadata from the collection.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager.
    batch_size : int, optional
        Number of records to process per batch.

    Returns
    -------
    int
        Number of collection records processed.
    """
    all_ids = _all_ids(embeddings_manager)
    processed = 0
    for batch in batched(all_ids, batch_size):
        existing = embeddings_manager.collection.get(ids=list(batch), include=["metadatas"])
        keys: set[str] = set()
        for metadata in existing.get("metadatas") or []:
            if metadata:
                keys.update(metadata)
        if keys:
            null_metadata = {key: None for key in keys}
            embeddings_manager.collection.update(ids=list(batch), metadatas=[null_metadata] * len(batch))
        processed += len(batch)
    return processed


def _update_documents(embeddings_manager: EmbeddingsManager, ids: list[str], documents: list[str]) -> None:
    """
    Update documents without touching embeddings or metadata.

    Uses ChromaDB's private update API because the public
    ``collection.update`` recomputes embeddings from documents when no
    embeddings are supplied.
    """
    collection = embeddings_manager.collection
    collection._client._update(  # type: ignore[attr-defined]
        collection_id=collection.id,
        ids=ids,
        embeddings=None,
        metadatas=None,
        documents=documents,
        tenant=collection.tenant,
        database=collection.database,
    )


def hash_all_documents(embeddings_manager: EmbeddingsManager, batch_size: int = _BATCH_SIZE) -> int:
    """
    Replace stored full-text documents with content hashes.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager.
    batch_size : int, optional
        Number of records to process per batch.

    Returns
    -------
    int
        Number of documents updated.
    """
    all_ids = _all_ids(embeddings_manager)
    updated = 0
    for batch in batched(all_ids, batch_size):
        existing = embeddings_manager.collection.get(ids=list(batch), include=["documents"])
        result_ids = existing.get("ids", [])
        result_documents = existing.get("documents", [])
        update_ids: list[str] = []
        update_documents: list[str] = []
        for uid, document in zip(result_ids, result_documents):
            if not document:
                continue
            new_document = str(document) if is_content_hash(document) else hash_text(str(document))
            if new_document == document:
                continue
            update_ids.append(uid)
            update_documents.append(new_document)
        if update_ids:
            _update_documents(embeddings_manager, update_ids, update_documents)
            updated += len(update_ids)
    return updated


def prune_orphan_embeddings(embeddings_manager: EmbeddingsManager, database: DatabaseManager) -> int:
    """
    Delete embeddings whose UIDs no longer exist in the paper database.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager.
    database : DatabaseManager
        Database used to resolve valid paper UIDs.

    Returns
    -------
    int
        Number of orphan embeddings deleted.
    """
    embedding_ids = set(_all_ids(embeddings_manager))
    paper_ids = set(database.get_paper_uids())
    return embeddings_manager.delete_embeddings_by_ids(list(embedding_ids - paper_ids))


def vacuum(embedding_db: str) -> None:
    """
    Reclaim disk space in a local ChromaDB directory.

    Parameters
    ----------
    embedding_db : str
        ChromaDB URL or local storage path.

    Raises
    ------
    RuntimeError
        If the database is remote and therefore cannot be vacuumed locally.
    """
    if embedding_db.startswith("http://") or embedding_db.startswith("https://"):
        raise RuntimeError("Cannot vacuum a remote ChromaDB instance from the client.")

    print("🧹 Vacuuming ChromaDB...")
    subprocess.run(
        ["chroma", "vacuum", "--path", str(Path(embedding_db)), "--force"],
        check=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the compaction migration.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
        get_config(reload=True)

    config = get_config()
    collection_name = args.collection or config.collection_name
    embeddings_manager = EmbeddingsManager(collection_name=collection_name)
    try:
        embeddings_manager.create_collection()
        print(f"🔧 Compacting collection '{collection_name}'...")

        metadata_records = remove_all_metadata(embeddings_manager)
        print(f"   Processed metadata for {metadata_records:,} record(s)")

        hashed_documents = hash_all_documents(embeddings_manager)
        print(f"   Hashed {hashed_documents:,} document(s)")

        if args.prune:
            with DatabaseManager() as database:
                pruned = prune_orphan_embeddings(embeddings_manager, database)
            print(f"   Pruned {pruned:,} orphan embedding(s)")
    except Exception as exc:
        print(f"❌ Compaction failed: {exc}", file=sys.stderr)
        return 1
    finally:
        embeddings_manager.close()

    if args.vacuum:
        try:
            vacuum(config.embedding_db)
        except RuntimeError as exc:
            print(f"⚠️  {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"❌ Vacuum failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
