#!/usr/bin/env python3
"""Update embedding model metadata in the paper database.

This script updates the embedding model stored in the
``embeddings_metadata`` table by using the existing database abstraction.
"""

from __future__ import annotations

import argparse
import os
import sys

from abstracts_explorer.database import DatabaseError, DatabaseManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Update the embedding model stored in database metadata. " "Optionally override PAPER_DB for this run."
        )
    )
    parser.add_argument(
        "embedding_model",
        help="New embedding model name to store in metadata.",
    )
    parser.add_argument(
        "--paper-db",
        help=("Override PAPER_DB for this command only. " "Accepts a SQLite path or full database URL."),
    )
    return parser.parse_args()


def main() -> int:
    """Run the script.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db

    try:
        with DatabaseManager() as db:
            previous_model = db.get_embedding_model()
            db.set_embedding_model(args.embedding_model)
            updated_model = db.get_embedding_model()

        print(f"Updated embedding model metadata: {previous_model!r} -> {updated_model!r}")
        return 0
    except DatabaseError as exc:
        print(f"Failed to update embedding model metadata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
