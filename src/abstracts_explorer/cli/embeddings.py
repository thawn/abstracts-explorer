"""
``create-embeddings`` command

"""

import argparse
import logging
import sys
from contextlib import closing


from tqdm import tqdm

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager, EmbeddingsError


from abstracts_explorer.cli._common import (
    _build_embeddings_where_filters,
    _print_header,
    _resolve_conference_arg,
    _unexpected_error,
)

logger = logging.getLogger(__name__)


def create_embeddings_command(args: argparse.Namespace) -> int:
    """
    Create embeddings database for abstracts.

    Without additional arguments, embeds all conference data for all
    available years.  With ``--conference`` and/or ``--year``, only the
    matching papers are embedded.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name for the ChromaDB collection
        - backend_url: URL for OpenAI-compatible API
        - model: Name of the embedding model
        - force: Whether to reset existing collection
        - conference: Conference name to filter by (optional)
        - year: Year to filter by (optional)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Resolve conference name once to canonical form
    conference = _resolve_conference_arg(getattr(args, "conference", None))
    args.conference = conference

    # Build structured paper filters from --conference and --year
    where_filters = _build_embeddings_where_filters(args)

    rate_limit_str = f"{args.requests_per_minute} req/min" if args.requests_per_minute > 0 else "disabled"
    year = getattr(args, "year", None)
    _print_header(
        "Embeddings Generator",
        f"Database: {config.masked_database_url}",
        f"Embedding DB: {config.embedding_db}",
        f"Collection: {args.collection}",
        f"Model: {args.model}",
        f"API URL: {args.backend_url}",
        f"Rate limit: {rate_limit_str}",
        f"Conference: {conference}" if conference else None,
        f"Year: {year}" if year is not None else None,
    )

    # Check paper count
    with DatabaseManager() as db:
        total_papers = db.get_paper_count()
        print(f"\n📊 Found {total_papers:,} papers in database")

        if where_filters:
            filtered_count = db.get_total_presentation_count(where_filters=where_filters)
            print(f"📊 Filter will process {filtered_count:,} abstracts")

    # Initialize embeddings manager
    try:
        print("\n🔧 Initializing embeddings manager...")
        with closing(
            EmbeddingsManager(
                backend_url=args.backend_url,
                model_name=args.model,
                collection_name=args.collection,
                requests_per_minute=args.requests_per_minute,
            )
        ) as em:
            # Check for model mismatch
            print("🔍 Checking embedding model compatibility...")
            compatible, stored_model, current_model = em.check_model_compatibility()

            if not compatible:
                print("\n⚠️  WARNING: Embedding model mismatch detected!")
                print(f"   Stored model:  {stored_model}")
                print(f"   Current model: {current_model}")
                print("\n   Embeddings created with different models are incompatible.")

                if args.force:
                    print("   --force flag detected: will recreate embeddings with new model")
                elif getattr(args, "ignore_embedding_model_mismatch", False):
                    print("   --ignore-embedding-model-mismatch detected: " "will recreate embeddings with new model")
                    args.force = True
                elif getattr(args, "non_interactive", False):
                    print(
                        "\n   Re-run with --ignore-embedding-model-mismatch to recreate the "
                        "embeddings without prompting.",
                        file=sys.stderr,
                    )
                    return 1
                else:
                    print("\n   Use --force to recreate embeddings with the new model.")
                    print("   This will delete existing embeddings and recreate them.\n")
                    response = input("Do you want to recreate embeddings with the new model? (y/N): ")
                    if response.lower() not in ["y", "yes"]:
                        print("\n❌ Aborted by user. Use --force to skip this prompt.")
                        return 1
                    # User confirmed, set force flag
                    args.force = True

            # Test connection
            if not em.test_api_connection(verbose=True):
                return 1

            # Create or reset collection.
            # When --force is combined with --conference / --year, only delete the
            # embeddings for the matching subset rather than wiping the entire
            # collection (which would lose embeddings for other conferences/years).
            filtered_force = args.force and (conference is not None or year is not None)
            if args.force and not filtered_force:
                print(f"🔄 Resetting existing collection '{args.collection}'...")
            elif filtered_force:
                scope_parts = []
                if conference:
                    scope_parts.append(f"conference={conference}")
                if year is not None:
                    scope_parts.append(f"year={year}")
                print(
                    f"🔄 Removing existing embeddings for {', '.join(scope_parts)} from collection '{args.collection}'..."
                )
            else:
                print(f"📁 Creating collection '{args.collection}'...")

            em.create_collection(reset=args.force and not filtered_force)

            if filtered_force:
                with DatabaseManager() as db:
                    existing_ids = db.get_paper_uids(where_filters)
                deleted = em.delete_embeddings_by_ids(existing_ids)
                if deleted:
                    print(f"   Removed {deleted:,} existing embeddings")

            # Generate embeddings with progress bar
            print("\n🚀 Generating embeddings...")

            # Determine total count for progress bar
            with DatabaseManager() as db:
                if where_filters:
                    total_count = db.get_total_presentation_count(where_filters=where_filters)
                else:
                    total_count = db.get_total_presentation_count()

            # Create progress bar
            with tqdm(total=total_count, desc="Embedding abstracts", unit="papers") as pbar:

                def update_progress(current: int, total: int) -> None:
                    pbar.n = current
                    pbar.total = total
                    pbar.refresh()

                embedded_count = em.embed_from_database(
                    where_filters=where_filters,
                    progress_callback=update_progress,
                    force_recreate=args.force,
                )

            print(f"✅ Successfully generated embeddings for {embedded_count:,} abstracts")

            # Show collection stats
            stats = em.get_collection_stats()
            print("\n📊 Collection Statistics:")
            print(f"   Name:  {stats['name']}")
            print(f"   Count: {stats['count']:,} documents")

        print(f"\n💾 Vector database saved to: {config.embedding_db}")
        print("\nYou can now use the 'search' command or the search_similar() method to find relevant abstracts!")

        return 0

    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)
