"""
Command-line interface for neurips-abstracts package.

This module provides CLI commands for Abstracts Explorer,
including downloading data, creating databases, and generating embeddings.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import argcomplete

from tqdm import tqdm

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager, EmbeddingsError
from abstracts_explorer.clustering import perform_clustering, compute_clusters_with_cache, ClusteringError
from abstracts_explorer.rag import RAGChat, RAGError
from abstracts_explorer.plugins import (
    DownloaderPlugin,
    LightweightPaper,
    get_plugin,
    get_all_plugins,
    list_plugins,
    list_plugin_names,
)
from abstracts_explorer.mcp_server import run_mcp_server
from abstracts_explorer.evaluation import (
    EvaluationError,
    Evaluator,
    format_eval_summary,
    format_eval_result_detail,
)

try:
    from abstracts_explorer._version import __version__
except ImportError:
    from abstracts_explorer import __version__  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    """
    Configure logging based on verbosity level and LOG_LEVEL environment variable.

    Precedence (highest to lowest):
    1. Command-line verbosity flags (-v, -vv)
    2. LOG_LEVEL environment variable
    3. Default (WARNING)

    Parameters
    ----------
    verbosity : int
        Verbosity level from command-line flags:
        - 0: Use LOG_LEVEL env var or WARNING (default)
        - 1: INFO
        - 2+: DEBUG
    """
    # Start with default level
    level = logging.WARNING

    # Check if verbosity flags were used
    if verbosity == 0:
        # No verbosity flags - check environment variable
        config = get_config()
        if config.log_level:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            level = level_map.get(config.log_level, logging.WARNING)
        # else: level remains at default WARNING (set above)
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Force reconfiguration even if logging is already configured
    )
    # Reset the package logger so it inherits the newly configured root level.
    # This overrides the level set at import time by _configure_package_logging().
    logging.getLogger("abstracts_explorer").setLevel(logging.NOTSET)


def add_conference_year_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard --conference and --year arguments to a parser.

    Creates a unified, case-insensitive option handler for conference and year
    filtering.  All commands that accept these options should use this helper
    so that help text, types, and defaults are consistent across the CLI.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.
    """
    parser.add_argument(
        "--conference",
        type=str,
        default=None,
        help="Conference to use (default: all conferences). Case-insensitive.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year of conference/workshop (default: all available years).",
    )


def _resolve_conference_arg(conference: Optional[str]) -> Optional[str]:
    """
    Resolve a CLI ``--conference`` value to the canonical conference name.

    Opens the local database, calls
    :py:meth:`~abstracts_explorer.database.DatabaseManager.resolve_conference_name`,
    and returns the result.  If *conference* is ``None`` or the database
    cannot be reached, the original value is returned unchanged.

    This helper should be called **once** at the entry point of every CLI
    command that accepts a ``--conference`` option.  After the call, all
    subsequent operations within that command receive the exact name as
    stored in the database (or the plugin's canonical name), preventing
    case-mismatch errors.

    Parameters
    ----------
    conference : str or None
        Raw value from ``args.conference``.

    Returns
    -------
    str or None
        Resolved conference name, or ``None`` if *conference* was ``None``.
    """
    if not conference:
        return conference
    with DatabaseManager() as db:
        db.create_tables()
        resolved_conference = db.resolve_conference_name(conference)
    if resolved_conference is not None and resolved_conference != conference:
        print(f"ℹ️  Resolved conference '{conference}' → '{resolved_conference}'")
    return resolved_conference


def _build_embeddings_where_clause(args: argparse.Namespace) -> Optional[str]:
    """
    Build a SQL WHERE clause from --conference, --year, and --where arguments.

    The conference name in *args* is expected to already be resolved to its
    canonical form (via ``DatabaseManager.resolve_conference_name``); an exact
    comparison is used.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments that may contain conference, year, and where attributes.

    Returns
    -------
    str or None
        Combined WHERE clause string, or None if no filters specified.
    """
    conditions = []
    conference = getattr(args, "conference", None)
    year = getattr(args, "year", None)
    where = getattr(args, "where", None)

    if conference:
        # Escape single quotes; conference name is already resolved to canonical form
        safe_conf = conference.replace("'", "''")
        conditions.append(f"conference = '{safe_conf}'")
    if year is not None:
        conditions.append(f"year = {int(year)}")
    if where:
        conditions.append(f"({where})")

    return " AND ".join(conditions) if conditions else None


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
        - lm_studio_url: URL for OpenAI-compatible API
        - model: Name of the embedding model
        - force: Whether to reset existing collection
        - conference: Conference name to filter by (optional)
        - year: Year to filter by (optional)
        - where: SQL WHERE clause to filter papers (optional)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Resolve conference name once to canonical form
    conference = _resolve_conference_arg(getattr(args, "conference", None))

    # Build combined WHERE clause from --conference, --year, and --where
    where_clause = _build_embeddings_where_clause(args)

    print("Abstracts Explorer - Embeddings Generator")
    print("=" * 70)
    print(f"Database: {config.database_url}")
    print(f"Embedding DB:   {config.embedding_db}")
    print(f"Collection: {args.collection}")
    print(f"Model:    {args.model}")
    print(f"API URL: {args.lm_studio_url}")
    rate_limit_str = f"{args.requests_per_minute} req/min" if args.requests_per_minute > 0 else "disabled"
    print(f"Rate limit: {rate_limit_str}")
    year = getattr(args, "year", None)
    if conference:
        print(f"Conference: {conference}")
    if year is not None:
        print(f"Year:       {year}")
    print("=" * 70)

    # Check paper count
    with DatabaseManager() as db:
        total_papers = db.get_paper_count()
        print(f"\n📊 Found {total_papers:,} papers in database")

        if where_clause:
            # Count papers matching filter
            filtered = db.query(f"SELECT COUNT(*) as count FROM papers WHERE {where_clause}")
            filtered_count = filtered[0]["count"] if filtered else 0
            print(f"📊 Filter will process {filtered_count:,} abstracts")

    # Initialize embeddings manager
    try:
        print("\n🔧 Initializing embeddings manager...")
        em = EmbeddingsManager(
            lm_studio_url=args.lm_studio_url,
            model_name=args.model,
            collection_name=args.collection,
            requests_per_minute=args.requests_per_minute,
        )

        # Check for model mismatch
        print("🔍 Checking embedding model compatibility...")
        compatible, stored_model, current_model = em.check_model_compatibility()

        if not compatible:
            print("\n⚠️  WARNING: Embedding model mismatch detected!")
            print(f"   Stored model:  {stored_model}")
            print(f"   Current model: {current_model}")
            print("\n   Embeddings created with different models are incompatible.")

            if not args.force:
                print("\n   Use --force to recreate embeddings with the new model.")
                print("   This will delete existing embeddings and recreate them.\n")
                response = input("Do you want to recreate embeddings with the new model? (y/N): ")
                if response.lower() not in ["y", "yes"]:
                    print("\n❌ Aborted by user. Use --force to skip this prompt.")
                    return 1
                # User confirmed, set force flag
                args.force = True
            else:
                print("   --force flag detected: will recreate embeddings with new model")

        # Test connection
        print("🔌 Testing OpenAI API connection...")
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - OpenAI-compatible API is running at {args.lm_studio_url}", file=sys.stderr)
            print(f"  - The {args.model} model is loaded", file=sys.stderr)
            return 1
        print("✅ Successfully connected to OpenAI API\n")

        # Connect to ChromaDB
        em.connect()

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
            deleted = em.delete_embeddings_by_filter(conference=conference, year=year)
            if deleted:
                print(f"   Removed {deleted:,} existing embeddings")

        # Generate embeddings with progress bar
        print("\n🚀 Generating embeddings...")

        # Determine total count for progress bar
        with DatabaseManager() as db:
            if where_clause:
                count_result = db.query(f"SELECT COUNT(*) as count FROM papers WHERE {where_clause}")
                total_count = count_result[0]["count"] if count_result else 0
            else:
                total_count = db.get_paper_count()

        # Create progress bar
        with tqdm(total=total_count, desc="Embedding abstracts", unit="papers") as pbar:

            def update_progress(current: int, total: int) -> None:
                pbar.n = current
                pbar.total = total
                pbar.refresh()

            embedded_count = em.embed_from_database(
                where_clause=where_clause,
                progress_callback=update_progress,
                force_recreate=args.force,
            )

        print(f"✅ Successfully generated embeddings for {embedded_count:,} abstracts")

        # Show collection stats
        stats = em.get_collection_stats()
        print("\n📊 Collection Statistics:")
        print(f"   Name:  {stats['name']}")
        print(f"   Count: {stats['count']:,} documents")

        em.close()

        print(f"\n💾 Vector database saved to: {config.embedding_db}")
        print("\nYou can now use the 'search' command or the search_similar() method to find relevant abstracts!")

        return 0

    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def search_command(args: argparse.Namespace) -> int:
    """
    Search the vector database for similar papers.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - query: Search query text
        - collection: Name of the ChromaDB collection
        - n_results: Number of results to return
        - where: Metadata filter conditions
        - show_abstract: Whether to show paper abstracts
        - lm_studio_url: URL for OpenAI-compatible API
        - model: Name of the embedding model

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not config.embedding_db.startswith("http://") and not config.embedding_db.startswith("https://"):
        embeddings_path = Path(config.embedding_db)
        if not embeddings_path.exists():
            print(f"❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
            print("\nYou can create embeddings using:", file=sys.stderr)
            print("  abstracts-explorer create-embeddings", file=sys.stderr)
            return 1

    # Resolve conference name once to canonical form
    conference = _resolve_conference_arg(getattr(args, "conference", None))
    year = getattr(args, "year", None)
    distance_threshold = getattr(args, "distance_threshold", 1.2)

    print("NeurIPS Semantic Search")
    print("=" * 70)
    print(f"Query: {args.query}")
    print(f"Embeddings: {config.embedding_db}")
    print(f"Collection: {args.collection}")
    print(f"Results: {args.n_results}")
    print(f"Distance threshold: {distance_threshold}")
    if conference:
        print(f"Conference: {conference}")
    if year is not None:
        print(f"Year:       {year}")
    print("=" * 70)

    try:
        # Initialize embeddings manager
        em = EmbeddingsManager(
            lm_studio_url=args.lm_studio_url,
            model_name=args.model,
            collection_name=args.collection,
        )

        # Test OpenAI API connection
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - OpenAI-compatible API is running at {args.lm_studio_url}", file=sys.stderr)
            print(f"  - The {args.model} model is loaded", file=sys.stderr)
            return 1

        # Connect to ChromaDB
        em.connect()
        em.create_collection()

        # Check collection stats
        stats = em.get_collection_stats()
        print(f"\n📊 Searching {stats['count']:,} papers in collection '{stats['name']}'")

        # Parse metadata filter from --where option
        where_filter: Optional[Dict[str, Any]] = None
        if args.where:
            # Simple key=value parsing
            try:
                where_filter = {}
                for condition in args.where.split(","):
                    key, value = condition.split("=", 1)
                    where_filter[key.strip()] = value.strip().strip("\"'")
                print(f"🔍 Filter: {where_filter}")
            except Exception as e:
                print(f"⚠️  Warning: Could not parse filter '{args.where}': {e}", file=sys.stderr)

        # Add conference/year filters from dedicated options
        if conference or year is not None:
            if where_filter is None:
                where_filter = {}
            if conference:
                # Conference name is already resolved to canonical form
                where_filter["conference"] = conference
            if year is not None:
                # ChromaDB stores all metadata values as strings
                where_filter["year"] = str(year)

        # Perform search
        print(f"\n🔍 Searching for: '{args.query}'...\n")
        results = em.search_similar(
            query=args.query,
            n_results=args.n_results,
            where=where_filter,
        )

        # Display results
        if not results["ids"] or not results["ids"][0]:
            print("❌ No results found.")
            em.close()
            return 0

        # Filter by distance threshold
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results.get("documents", [[]])[0]

        filtered = [
            (ids[i], metadatas[i], distances[i], documents[i] if documents else "")
            for i in range(len(ids))
            if distances[i] <= distance_threshold
        ]

        if not filtered:
            print(f"❌ No results found within distance threshold {distance_threshold}.")
            em.close()
            return 0

        num_results = len(filtered)
        print(f"✅ Found {num_results} similar paper(s):\n")

        for i, (paper_id, metadata, distance, document) in enumerate(filtered):
            similarity = 1 - distance if distance <= 1 else 0

            print(f"{i + 1}. Paper ID: {paper_id}")
            print(f"   Title: {metadata.get('title', 'N/A')}")

            # Get author names from metadata
            authors_display = metadata.get("authors", "N/A")

            print(f"   Authors: {authors_display}")
            print(f"   Decision: {metadata.get('decision', 'N/A')}")

            if metadata.get("topic"):
                print(f"   Topic: {metadata.get('topic')}")

            if metadata.get("paper_url"):
                print(f"   URL: {metadata.get('paper_url')}")

            if metadata.get("poster_position"):
                print(f"   Poster Position: {metadata.get('poster_position')}")

            print(f"   Similarity: {similarity:.4f}")

            if args.show_abstract and document:
                abstract = document[:300] + "..." if len(document) > 300 else document
                print(f"   Abstract: {abstract}")

            print()

        em.close()
        return 0

    except EmbeddingsError as e:
        print(f"\n❌ Search error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def chat_command(args: argparse.Namespace) -> int:
    """
    Interactive RAG chat with papers.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the collection
        - lm_studio_url: LM Studio API URL
        - model: Language model name for chat
        - embedding_model: Embedding model name
        - max_context: Maximum papers to use as context
        - temperature: Sampling temperature
        - export: Path to export conversation
        - conference: Conference to scope chat context (optional)
        - year: Year to scope chat context (optional)

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    try:
        config = get_config()

        # Resolve conference name once to canonical form
        conference = _resolve_conference_arg(getattr(args, "conference", None))
        year = getattr(args, "year", None)

        print("=" * 70)
        print("NeurIPS RAG Chat")
        print("=" * 70)
        print(f"Embeddings: {config.embedding_db}")
        print(f"Collection: {args.collection}")
        print(f"Chat Model: {args.model}")
        print(f"Embedding Model: {args.embedding_model}")
        print(f"API URL: {args.lm_studio_url}")
        if conference:
            print(f"Conference: {conference}")
        if year is not None:
            print(f"Year:       {year}")
        print("=" * 70)

        # Check embeddings exist (for local paths only)
        if not config.embedding_db.startswith("http://") and not config.embedding_db.startswith("https://"):
            embeddings_path = Path(config.embedding_db)
            if not embeddings_path.exists():
                print(f"\n❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
                print("\nYou can create embeddings using:", file=sys.stderr)
                print("  abstracts-explorer create-embeddings", file=sys.stderr)
                return 1

        # Initialize embeddings manager
        em = EmbeddingsManager(
            collection_name=args.collection,
            lm_studio_url=args.lm_studio_url,
            model_name=args.embedding_model,
        )

        # Test OpenAI API connection
        print("\n🔌 Testing OpenAI API connection...")
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - OpenAI-compatible API is running at {args.lm_studio_url}", file=sys.stderr)
            print("  - A language model is loaded", file=sys.stderr)
            return 1
        print("✅ Successfully connected to OpenAI API")

        # Connect to embeddings
        em.connect()

        # Get or create the collection (should already exist for chat)
        em.create_collection(reset=False)

        # Get collection stats
        stats = em.get_collection_stats()
        print(f"\n📊 Loaded {stats['count']:,} papers from collection '{stats['name']}'")

        # Initialize database connection
        db = DatabaseManager()
        db.connect()

        # Build conference/year context for RAG queries (case-insensitive via DB lookup)
        chat_conferences: Optional[List[str]] = None
        chat_years: Optional[List[int]] = None
        if conference:
            rows = db.query(
                "SELECT DISTINCT conference FROM papers WHERE LOWER(conference) = LOWER(?)",
                (conference,),
            )
            stored_name = rows[0]["conference"] if rows else conference
            chat_conferences = [stored_name]
        if year is not None:
            chat_years = [year]

        # Initialize RAG chat
        chat = RAGChat(
            embeddings_manager=em,
            database=db,
            lm_studio_url=args.lm_studio_url,
            model=args.model,
            max_context_papers=args.max_context,
            temperature=args.temperature,
        )

        print("\n💬 Chat started! Type 'exit' or 'quit' to end, 'reset' to clear history.")
        print("=" * 70)
        print()

        # Interactive chat loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "reset":
                    chat.reset_conversation()
                    print("🔄 Conversation history cleared.\n")
                    continue

                if user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  exit, quit, q  - Exit chat")
                    print("  reset          - Clear conversation history")
                    print("  help           - Show this help message")
                    print()
                    continue

                # Query RAG system
                print("\n🔍 Searching papers...", end="", flush=True)
                result = chat.query(user_input, conferences=chat_conferences, years=chat_years)
                print("\r" + " " * 50 + "\r", end="")  # Clear the line

                # Display response
                print(f"Assistant (based on {result['metadata']['n_papers']} papers):")
                print(result["response"])
                print()

                # Show source papers if requested
                if args.show_sources and result["papers"]:
                    print("📚 Source papers:")
                    for i, paper in enumerate(result["papers"], 1):
                        score = paper.get("relevance_score", paper.get("similarity", 0))
                        print(f"  {i}. {paper.get('title', 'Unknown')} (relevance: {score:.3f})")
                    print()

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break

        # Export conversation if requested
        if args.export:
            export_path = Path(args.export)
            chat.export_conversation(export_path)
            print(f"💾 Conversation exported to: {export_path}")

        db.close()
        em.close()
        return 0

    except RAGError as e:
        print(f"\n❌ RAG error: {e}", file=sys.stderr)
        return 1
    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def _download_single(
    plugin: DownloaderPlugin,
    year: int,
    output_path: Path,
    force: bool,
    args: argparse.Namespace,
) -> List[LightweightPaper]:
    """
    Download papers for a single plugin and year.

    Parameters
    ----------
    plugin : DownloaderPlugin
        The plugin instance to use for downloading
    year : int
        Conference year to download
    output_path : Path
        Base output path for intermediate JSON files
    force : bool
        Whether to force re-download
    args : argparse.Namespace
        Original CLI arguments (for plugin-specific options)

    Returns
    -------
    list of LightweightPaper
        Downloaded papers

    Raises
    ------
    Exception
        If the download fails
    """
    kwargs: Dict[str, Any] = {}

    if plugin.plugin_name == "ml4ps":
        kwargs["max_workers"] = getattr(args, "max_workers", 20)

    if plugin.plugin_name == "chi":
        input_file = getattr(args, "input_file", None)
        if input_file:
            kwargs["input_path"] = input_file

    json_path = output_path.parent / f"{plugin.plugin_name}_{year}.json"
    return plugin.download(year=year, output_path=str(json_path), force_download=force, **kwargs)


def download_command(args: argparse.Namespace) -> int:
    """
    Download conference data and create database.

    Without additional arguments, downloads all available conferences
    for all supported years.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - conference: Name of the downloader plugin to use (None for all)
        - year: Year of conference (None for all supported years)
        - output: Path for intermediate JSON files
        - force: Whether to force re-download
        - list_plugins: Whether to list available plugins

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    # Import plugins to register them

    # If list_plugins flag is set, show available plugins and exit
    if hasattr(args, "list_plugins") and args.list_plugins:
        print("Available downloader plugins:")
        print("=" * 70)
        plugins = list_plugins()
        for plugin_meta in plugins:
            print(f"\n📦 {plugin_meta['name']}")
            print(f"   {plugin_meta['description']}")
            if plugin_meta.get("supported_years"):
                years = plugin_meta["supported_years"]
                if len(years) > 5:
                    print(f"   Supported years: {min(years)}-{max(years)}")
                else:
                    print(f"   Supported years: {', '.join(map(str, years))}")
        print("\n" + "=" * 70)
        return 0

    output_path = Path(args.output)
    plugin_name = getattr(args, "conference", None)

    # Determine which plugins to use
    if plugin_name:
        # Specific plugin requested – look up case-insensitively (plugin names are lowercase)
        plugin = get_plugin(plugin_name.lower())
        if not plugin:
            print(f"❌ Error: Plugin '{plugin_name}' not found", file=sys.stderr)
            print(f"\nAvailable plugins: {', '.join(list_plugin_names())}", file=sys.stderr)
            print("\nUse --list-plugins to see details", file=sys.stderr)
            return 1
        plugins_to_download = [plugin]
    else:
        # No plugin specified: download all plugins
        plugins_to_download = list(get_all_plugins())

    total_papers = 0
    errors = []

    for plugin in plugins_to_download:
        # Determine years to download
        year = getattr(args, "year", None)
        if year is not None:
            years = [year]
        else:
            years = sorted(plugin.supported_years) if plugin.supported_years else []

        if not years:
            print(f"⚠️  No supported years for {plugin.plugin_name}, skipping.")
            continue

        print(f"Using plugin: {plugin.plugin_description}")
        print(f"Downloading {plugin.plugin_name} for year(s): {', '.join(map(str, years))}...")
        print("=" * 70)

        for yr in years:
            try:
                papers = _download_single(plugin, yr, output_path, args.force, args)
                print(f"✅ Downloaded {len(papers):,} papers for {plugin.plugin_name} {yr}")

                with DatabaseManager() as db:
                    db.create_tables()
                    count = db.add_papers(papers)
                    print(f"✅ Loaded {count:,} papers into database")

                total_papers += len(papers)

            except Exception as e:
                msg = f"{plugin.plugin_name} {yr}: {e}"
                errors.append(msg)
                print(f"❌ Error downloading {plugin.plugin_name} {yr}: {e}", file=sys.stderr)
                import traceback

                if args.verbose > 0:
                    traceback.print_exc()

        print()

    config = get_config()
    print(f"💾 Database updated: {config.database_url}")
    print(f"📊 Total papers downloaded: {total_papers:,}")

    if errors:
        print(f"\n⚠️  {len(errors)} error(s) occurred:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        return 1

    return 0


def web_ui_command(args: argparse.Namespace) -> int:
    """
    Start the web UI server.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - host: Host to bind to
        - port: Port to bind to
        - verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        - dev: Use Flask development server
        - threads: Number of Waitress worker threads

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Try to import Flask - if it fails, give helpful error
        try:
            from abstracts_explorer.web_ui import run_server
        except ImportError:
            print("\n❌ Web UI dependencies not installed!", file=sys.stderr)
            print("\nThe web UI requires Waitress and Flask. Install them with:", file=sys.stderr)
            print("  uv sync --extra web", file=sys.stderr)
            print("\nOr install Flask manually:", file=sys.stderr)
            print("  pip install flask flask-cors", file=sys.stderr)
            return 1

        # Determine debug mode from verbosity level (2+ = DEBUG)
        debug = getattr(args, "verbose", 0) >= 2

        # Start the server (dev defaults to False for production server)
        run_server(
            host=args.host,
            port=args.port,
            debug=debug,
            dev=getattr(args, "dev", False),
            threads=getattr(args, "threads", 6),
        )
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except ValueError as e:
        print(f"\n❌ Invalid configuration: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        # Database not found - error message already printed by run_server
        return 1
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cluster_embeddings_command(args: argparse.Namespace) -> int:
    """
    Cluster embeddings and optionally export results.

    Uses agglomerative clustering with ward linkage, distance threshold 150,
    and t-SNE dimensionality reduction.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - output: Path to export JSON results
        - limit: Maximum number of embeddings to process

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not config.embedding_db.startswith("http://") and not config.embedding_db.startswith("https://"):
        embeddings_path = Path(config.embedding_db)
        if not embeddings_path.exists():
            print(f"❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
            print("\nYou can create embeddings using:", file=sys.stderr)
            print("  abstracts-explorer create-embeddings", file=sys.stderr)
            return 1

    print("Abstracts Explorer - Clustering")
    print("=" * 70)
    print(f"Embeddings:  {config.embedding_db}")
    print(f"Collection:  {args.collection}")
    print("Reduction:   tsne (n_components=2)")
    print("Clustering:  agglomerative (linkage=ward, distance_threshold=150)")

    if args.limit:
        print(f"Limit:       {args.limit} papers")
    print("=" * 70)

    try:
        # Perform clustering with fixed parameters
        print("\n🚀 Starting clustering pipeline...")
        results = perform_clustering(
            collection_name=args.collection,
            reduction_method="tsne",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            output_path=args.output,
            limit=args.limit,
            distance_threshold=150.0,
            linkage="ward",
        )

        # Display statistics
        stats = results["statistics"]
        print("\n📊 Clustering Results:")
        print(f"   Total papers:  {stats['total_papers']:,}")
        print(f"   Clusters:      {stats['n_clusters']}")
        if stats["n_noise"] > 0:
            print(f"   Noise points:  {stats['n_noise']}")
        print("\n   Cluster sizes:")
        for cluster_id, size in sorted(stats["cluster_sizes"].items()):
            print(f"      Cluster {cluster_id}: {size:,} papers")

        if args.output:
            print(f"\n💾 Results exported to: {args.output}")
            print("\nYou can use the web UI to visualize the clusters!")
        else:
            print("\n💡 Tip: Use --output to export results for visualization")

        return 0

    except ClusteringError as e:
        print(f"\n❌ Clustering error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def clear_clustering_cache_command(args: argparse.Namespace) -> int:
    """
    Clear clustering cache from the database.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - embedding_model: Optional embedding model to filter by

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    print("Abstracts Explorer - Clear Clustering Cache")
    print("=" * 70)
    print(f"Database: {config.database_url}")
    if args.embedding_model:
        print(f"Filtering by model: {args.embedding_model}")
    else:
        print("Clearing all cache entries")
    print("=" * 70)

    try:
        # Connect to database
        with DatabaseManager() as db:
            # Clear the cache
            count = db.clear_clustering_cache(embedding_model=args.embedding_model)

            if count == 0:
                print("\n✅ No cache entries found to clear.")
            else:
                entry_word = "entry" if count == 1 else "entries"
                if args.embedding_model:
                    print(f"\n✅ Cleared {count} clustering cache {entry_word} for model: {args.embedding_model}")
                else:
                    print(f"\n✅ Cleared all {count} clustering cache {entry_word}")

        return 0

    except Exception as e:
        print(f"\n❌ Error clearing clustering cache: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


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
        print("❌ --conference is required for the delete-data command.", file=sys.stderr)
        return 1
    if year is None:
        print("❌ --year is required for the delete-data command.", file=sys.stderr)
        return 1

    # Resolve conference name to canonical form
    try:
        conference = _resolve_conference_arg(conference)
    except Exception:
        # If resolution fails, use the raw value (might still be valid)
        pass

    print("Abstracts Explorer - Delete Conference/Year Data")
    print("=" * 70)
    print(f"Database:     {config.database_url}")
    print(f"Embedding DB: {config.embedding_db}")
    print(f"Conference:   {conference}")
    print(f"Year:         {year}")
    print("=" * 70)

    # Count what will be deleted
    paper_count = 0
    cache_count = 0
    try:
        with DatabaseManager() as db:
            db.create_tables()
            papers = db.search_papers(conference=conference, year=year, limit=0)
            paper_count = len(papers)
            cache_count = db.count_clustering_cache_by_conference_year(conference, year)
    except Exception as e:
        print(f"\n❌ Error accessing database: {e}", file=sys.stderr)
        return 1

    # Count ChromaDB embeddings
    embedding_count = 0
    try:
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=False)
        existing = em.collection.get(where={"$and": [{"conference": conference}, {"year": str(year)}]})
        embedding_count = len(existing.get("ids", []))
        em.close()
    except Exception as e:
        logger.warning(f"Could not count ChromaDB embeddings (will still attempt deletion): {e}")
        embedding_count = 0

    print("\nThe following data will be permanently deleted:")
    print(f"  📄 Papers:            {paper_count:,}")
    print(f"  🔢 Embeddings:        {embedding_count:,}")
    print(f"  🗂️  Clustering cache:  {cache_count:,}")

    if paper_count == 0 and embedding_count == 0 and cache_count == 0:
        print("\n✅ Nothing to delete — no data found for this conference/year combination.")
        return 0

    if not getattr(args, "yes", False):
        print(f"\n⚠️  This will permanently delete all data for {conference} {year} from all databases.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    errors: List[str] = []

    # 1. Delete papers from paper database
    try:
        with DatabaseManager() as db:
            deleted_papers = db.delete_papers_by_conference_year(conference, year)
        print(f"\n✅ Deleted {deleted_papers:,} paper(s) from paper database.")
    except Exception as e:
        msg = f"Failed to delete papers: {e}"
        print(f"\n❌ {msg}", file=sys.stderr)
        errors.append(msg)

    # 2. Delete embeddings from ChromaDB
    try:
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=False)
        deleted_embeddings = em.delete_embeddings_by_filter(conference=conference, year=year)
        em.close()
        print(f"✅ Deleted {deleted_embeddings:,} embedding(s) from ChromaDB.")
    except Exception as e:
        msg = f"Failed to delete embeddings: {e}"
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


def pre_generate_clustering_command(args: argparse.Namespace) -> int:
    """
    Pre-generate clustering results for one or all conference/year combinations.

    Uses agglomerative clustering with ward linkage, distance threshold 150,
    and t-SNE dimensionality reduction.  Results are persisted to the database
    cache so that the web UI can serve them instantly.

    Without ``--conference`` or ``--year``, generates clustering for every
    conference in the database combined with each individual year.

    With ``--conference`` only, generates for that conference for each individual year.

    With both ``--conference`` and ``--year``, generates for that specific
    conference + year only.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - conference: Single conference to filter by (optional)
        - year: Single year to filter by (optional)
        - force: Force recompute even if cache exists

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not config.embedding_db.startswith("http://") and not config.embedding_db.startswith("https://"):
        from pathlib import Path

        embeddings_path = Path(config.embedding_db)
        if not embeddings_path.exists():
            print(f"❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
            print("\nYou can create embeddings using:", file=sys.stderr)
            print("  abstracts-explorer create-embeddings", file=sys.stderr)
            return 1

    # Resolve conference name once to canonical form
    resolved_conference = _resolve_conference_arg(getattr(args, "conference", None))
    year_arg: Optional[int] = getattr(args, "year", None)

    # Build the list of (conference, year) combinations from the database
    combos: list = []
    if resolved_conference is not None and year_arg is not None:
        # Both specified: single combo, no DB lookup needed
        combos = [(resolved_conference, year_arg)]
    else:
        try:
            with DatabaseManager() as db:
                conferences = [resolved_conference] if resolved_conference else db.get_conferences()
                for conf in conferences:
                    for year in db.get_years(conference=conf):
                        combos.append((conf, year))
        except Exception:
            pass

    if not combos:
        print("❌ No conferences found in the database.", file=sys.stderr)
        return 1

    print("Abstracts Explorer - Pre-generate Clustering")
    print("=" * 70)
    print(f"Embeddings:       {config.embedding_db}")
    print(f"Collection:       {args.collection}")
    print("Clustering:       agglomerative (linkage=ward, distance_threshold=150)")
    print("Reduction:        tsne")
    rate_limit_str = f"{args.requests_per_minute} req/min" if args.requests_per_minute > 0 else "disabled"
    print(f"Rate limit:       {rate_limit_str}")
    if len(combos) > 1:
        print(f"Combinations:     {len(combos)} total")
    else:
        conf_name, yr = combos[0]
        if conf_name:
            print(f"Conference:       {conf_name}")
        if yr is not None:
            print(f"Year:             {yr}")
    print("=" * 70)

    try:
        em = EmbeddingsManager(
            collection_name=args.collection,
            requests_per_minute=args.requests_per_minute,
        )
        em.connect()
        em.create_collection()

        success_count = 0
        fail_count = 0

        for conf_name, yr in combos:
            c_conferences = [conf_name] if conf_name else None
            c_years = [yr] if yr is not None else None

            label = f"{conf_name or 'all'}"
            if yr is not None:
                label += f" {yr}"
            else:
                raise ValueError(
                    "Year must be specified for clustering pre-generation to ensure manageable computation"
                )

            print(f"\n🚀 Clustering {label}...")

            try:
                with DatabaseManager() as db:
                    results = compute_clusters_with_cache(
                        embeddings_manager=em,
                        database=db,
                        embedding_model=config.embedding_model,
                        reduction_method="tsne",
                        n_components=2,
                        clustering_method="agglomerative",
                        n_clusters=None,
                        limit=None,
                        force=args.force,
                        conferences=c_conferences,
                        years=c_years,
                        linkage="ward",
                        distance_threshold=150.0,
                    )

                stats = results.get("statistics", {})
                n_papers = stats.get("total_papers", "N/A")
                n_clusters = stats.get("n_clusters", "N/A")
                n_papers_str = f"{n_papers:,}" if isinstance(n_papers, int) else str(n_papers)
                print(f"   ✅ {label}: {n_papers_str} papers → {n_clusters} clusters")
                success_count += 1
            except Exception as e:
                print(f"   ❌ {label}: {e}")
                fail_count += 1

        print(f"\n{'=' * 70}")
        print(f"✅ Pre-generation complete: {success_count} succeeded, {fail_count} failed")
        return 0 if fail_count == 0 else 1

    except ClusteringError as e:
        print(f"\n❌ Clustering error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def mcp_server_command(args: argparse.Namespace) -> int:
    """
    Start the MCP server for cluster analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - host: Host to bind to
        - port: Port to bind to
        - transport: Transport method (sse or stdio)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        print("Abstracts Explorer - MCP Server")
        print("=" * 70)
        print(f"Host:      {args.host}")
        print(f"Port:      {args.port}")
        print(f"Transport: {args.transport}")
        print("=" * 70)
        print("\nStarting MCP server...")
        print("\nAvailable tools:")
        print("  - get_conference_topics: Get the main research topics of a conference")
        print("  - get_topic_evolution: Analyze how topics evolved over years")
        print("  - search_papers: Find abstracts about topics")
        print("  - get_cluster_visualization: Generate cluster visualization data")
        print("\nPress Ctrl+C to stop the server\n")

        # Start the MCP server
        run_mcp_server(host=args.host, port=args.port, transport=args.transport)
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting MCP server: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def eval_generate_command(args: argparse.Namespace) -> int:
    """
    Generate evaluation Q/A pairs using an LLM.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - n_pairs: Number of Q/A pairs per tool
        - tools: Specific tools to generate for (or all)
        - no_followups: Disable follow-up generation
        - n_followups: Number of follow-ups per pair
        - model: Chat model name
        - lm_studio_url: LLM backend URL

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    print("Abstracts Explorer - Generate Evaluation Q/A Pairs")
    print("=" * 70)
    print(f"Database:   {config.database_url}")
    print(f"Model:      {args.model}")
    print(f"Pairs/tool: {args.n_pairs}")
    print(f"Follow-ups: {'disabled' if args.no_followups else args.n_followups}")
    if args.tools:
        print(f"Tools:      {', '.join(args.tools)}")
    else:
        print("Tools:      all")
    print("=" * 70)

    try:
        # Initialize embeddings manager for its openai_client
        em = EmbeddingsManager(
            lm_studio_url=args.lm_studio_url,
        )

        with DatabaseManager() as db:
            db.create_tables()

            total_papers = db.get_paper_count()
            if total_papers == 0:
                print("\n❌ No papers in database. Download papers first.", file=sys.stderr)
                return 1
            print(f"\n📊 Found {total_papers:,} papers in database")

            evaluator = Evaluator(embeddings_manager=em, db=db, model=args.model)

            print("\n🤖 Generating Q/A pairs...")
            pairs = evaluator.generate_qa_pairs(
                n_pairs_per_tool=args.n_pairs,
                tools=args.tools if args.tools else None,
                generate_followups=not args.no_followups,
                n_followups=args.n_followups,
            )

            count = evaluator.store_qa_pairs(pairs)
            print(f"\n✅ Generated and stored {count} Q/A pair(s)")

            # Show summary by tool
            tool_counts: dict = {}
            for p in pairs:
                t = p.get("tool_name", "unknown")
                tool_counts[t] = tool_counts.get(t, 0) + 1
            print("\n📊 Pairs per tool:")
            for tool, cnt in sorted(tool_counts.items()):
                print(f"   {tool}: {cnt}")

        return 0

    except EvaluationError as e:
        print(f"\n❌ Evaluation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def eval_verify_command(args: argparse.Namespace) -> int:
    """
    Interactively verify, edit, or delete evaluation Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - sample: Number of pairs to sample for review
        - all: Review all unverified pairs

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            pairs = db.get_eval_qa_pairs(verified_only=False)
            if not pairs:
                print("No Q/A pairs found. Generate pairs first with 'eval generate'.")
                return 0

            # Filter to unverified pairs unless reviewing all
            if not args.all:
                unverified = [p for p in pairs if p["verified"] == 0]
            else:
                unverified = pairs

            if not unverified:
                print("✅ All pairs have been verified.")
                return 0

            # Sample if requested
            if args.sample and args.sample < len(unverified):
                import random

                review_set = random.sample(unverified, args.sample)
            else:
                review_set = unverified

            print(f"📝 Reviewing {len(review_set)} Q/A pair(s)")
            print("Commands: [a]ccept, [r]eject, [e]dit query, [E]dit answer, [s]kip, [h]elp, [q]uit\n")

            _VERIFY_HELP = (
                "  [a]  accept         — mark as verified\n"
                "  [r]  reject         — mark as rejected\n"
                "  [e]  edit query     — edit the query text, then accept\n"
                "  [E]  edit answer    — edit the expected answer, then accept\n"
                "  [s]  skip           — leave unchanged and move to the next pair\n"
                "  [h]  help           — show this help message\n"
                "  [q]  quit           — stop verification and show summary\n"
            )

            accepted = 0
            rejected = 0
            edited = 0
            quit_requested = False

            for i, pair in enumerate(review_set, 1):
                if quit_requested:
                    break
                print(f"--- Pair {i}/{len(review_set)} (ID: {pair['id']}) ---")
                print(f"Tool:     {pair.get('tool_name', 'N/A')}")
                print(f"Conv:     {pair['conversation_id']} turn {pair['turn_number']}")
                print(f"Query:    {pair['query']}")
                print(f"Answer:   {pair['expected_answer']}")
                status_map = {0: "unverified", 1: "verified", -1: "rejected"}
                print(f"Status:   {status_map.get(pair['verified'], 'unknown')}")

                action_taken = False
                while not action_taken:
                    try:
                        choice = input("\n[a/r/e/E/s/h/q]: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Verification stopped.")
                        quit_requested = True
                        action_taken = True
                        continue

                    if choice == "h":
                        print(_VERIFY_HELP)
                    elif choice in ("a", "A"):
                        db.update_eval_qa_pair(pair["id"], verified=1)
                        accepted += 1
                        print("✅ Accepted\n")
                        action_taken = True
                    elif choice in ("r", "R"):
                        db.update_eval_qa_pair(pair["id"], verified=-1)
                        rejected += 1
                        print("❌ Rejected\n")
                        action_taken = True
                    elif choice == "e":
                        try:
                            new_query = input("New query: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n\n👋 Verification stopped.")
                            quit_requested = True
                            action_taken = True
                            continue
                        if new_query:
                            db.update_eval_qa_pair(pair["id"], query=new_query, verified=1)
                            edited += 1
                            print("✏️  Updated and accepted\n")
                        else:
                            print("⏭️  Skipped (empty input)\n")
                        action_taken = True
                    elif choice in ("E", "ea"):
                        try:
                            new_answer = input("New answer: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n\n👋 Verification stopped.")
                            quit_requested = True
                            action_taken = True
                            continue
                        if new_answer:
                            db.update_eval_qa_pair(pair["id"], expected_answer=new_answer, verified=1)
                            edited += 1
                            print("✏️  Updated and accepted\n")
                        else:
                            print("⏭️  Skipped (empty input)\n")
                        action_taken = True
                    elif choice in ("q", "Q"):
                        print("👋 Quitting verification.")
                        quit_requested = True
                        action_taken = True
                    else:
                        print("⏭️  Skipped\n")
                        action_taken = True

            print(f"\n📊 Summary: {accepted} accepted, {rejected} rejected, {edited} edited")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def eval_run_command(args: argparse.Namespace) -> int:
    """
    Run evaluation on stored Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: ChromaDB collection name
        - model: Chat model name
        - lm_studio_url: LLM backend URL
        - embedding_model: Embedding model name
        - limit: Maximum pairs to evaluate
        - include_unverified: Include unverified pairs

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    print("Abstracts Explorer - Run Evaluation")
    print("=" * 70)
    print(f"Database:    {config.database_url}")
    print(f"Model:       {args.model}")
    print(f"Embeddings:  {config.embedding_db}")
    print(f"Collection:  {args.collection}")
    print("=" * 70)

    try:
        # Initialize embeddings
        em = EmbeddingsManager(
            lm_studio_url=args.lm_studio_url,
            model_name=args.embedding_model,
            collection_name=args.collection,
        )

        print("\n🔌 Testing OpenAI API connection...")
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
            return 1
        print("✅ Connected")

        em.connect()
        em.create_collection(reset=False)

        with DatabaseManager() as db:
            db.create_tables()

            pair_count = db.get_eval_qa_pair_count(verified_only=not args.include_unverified)
            if pair_count == 0:
                print("\n❌ No Q/A pairs found. Generate and verify pairs first.", file=sys.stderr)
                em.close()
                return 1

            actual_count = min(pair_count, args.limit) if args.limit else pair_count
            print(f"\n📊 Evaluating {actual_count} Q/A pair(s)...")

            evaluator = Evaluator(embeddings_manager=em, db=db, model=args.model)
            run_id = evaluator.run_evaluation(
                verified_only=not args.include_unverified,
                limit=args.limit,
            )

            # Show summary
            print(f"\n{evaluator.format_run_summary(run_id)}")

        em.close()
        return 0

    except EvaluationError as e:
        print(f"\n❌ Evaluation error: {e}", file=sys.stderr)
        return 1
    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def eval_results_command(args: argparse.Namespace) -> int:
    """
    Browse evaluation results.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - run_id: Specific run to show (or latest)
        - sample: Random sample size for browsing
        - detail: Show detailed per-pair results
        - clear: Delete results for the selected run
        - yes: Skip confirmation prompt when clearing

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            run_ids = db.get_eval_run_ids()
            if not run_ids:
                print("No evaluation results found. Run 'eval run' first.")
                return 0

            # Determine which run to show (runs are oldest-first, so [-1] is the most recent)
            if args.run_id:
                target_run = args.run_id
            else:
                target_run = run_ids[-1]
                print(f"Showing latest run: {target_run}")

            if target_run not in run_ids:
                print(f"❌ Run '{target_run}' not found.", file=sys.stderr)
                print(f"Available runs: {', '.join(run_ids)}")
                return 1

            # Handle --clear option
            if getattr(args, "clear", False):
                n = len(db.get_eval_results(run_id=target_run))
                if not getattr(args, "yes", False):
                    confirm = input(
                        f"⚠️  This will permanently delete {n} result(s) for run '{target_run}'. Are you sure? [y/N]: "
                    )
                    if confirm.strip().lower() != "y":
                        print("Aborted.")
                        return 0
                deleted = db.delete_eval_results(run_id=target_run)
                print(f"✅ Deleted {deleted} result(s) for run '{target_run}'.")
                return 0

            # Show summary
            summary = db.get_eval_run_summary(target_run)
            print(f"\n{format_eval_summary(summary, target_run)}")

            # Show details if requested
            if args.detail:
                results = db.get_eval_results(run_id=target_run)

                # Sample if requested
                if args.sample and args.sample < len(results):
                    import random

                    results = random.sample(results, args.sample)
                    print(f"\n📋 Showing random sample of {len(results)} result(s):\n")
                else:
                    print(f"\n📋 Showing all {len(results)} result(s):\n")

                # Build lookup for QA pairs
                pairs = db.get_eval_qa_pairs()
                pair_lookup = {p["id"]: p for p in pairs}

                for r in results:
                    qa = pair_lookup.get(r["qa_pair_id"])
                    print(format_eval_result_detail(r, qa))
                    print()

            # List all available runs (oldest first, so most recent is shown last)
            if len(run_ids) >= 1:
                print(f"\n📁 Available runs ({len(run_ids)}):")
                for rid in run_ids:
                    s = db.get_eval_run_summary(rid)
                    score_str = f"{s['avg_score']:.2f}" if s.get("avg_score") else "N/A"
                    run_date = s.get("run_date")
                    date_str = (
                        run_date.strftime("%Y-%m-%d %H:%M") if isinstance(run_date, datetime) else "unknown date"
                    )
                    marker = " ◀ current" if rid == target_run else ""
                    print(f"   {date_str}  {rid}: {s['total']} pairs, avg score: {score_str}{marker}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def eval_clear_command(args: argparse.Namespace) -> int:
    """
    Delete all accepted (verified) evaluation Q/A pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - yes: Skip confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            count = db.get_eval_qa_pair_count(verified_only=True)
            if count == 0:
                print("No accepted Q/A pairs found — nothing to clear.")
                return 0

            print(f"⚠️  This will permanently delete {count} accepted Q/A pair(s).")
            if not args.yes:
                answer = input("Are you sure? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Aborted.")
                    return 0

            deleted = db.delete_verified_eval_qa_pairs()
            print(f"✅ Deleted {deleted} accepted Q/A pair(s).")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def feedback_stats_command(args: argparse.Namespace) -> int:
    """
    Show summary statistics for user feedback (chat donations and data donations).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments (no extra fields required).

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            chat_stats = db.get_chat_donation_stats()
            val_stats = db.get_validation_data_stats()

            print("\n📊 Feedback Statistics")
            print("=" * 40)

            print("\n💬 Chat Donations")
            print(f"   Total:      {chat_stats['total']}")
            print(f"   👍 Up:       {chat_stats['up']}")
            print(f"   👎 Down:     {chat_stats['down']}")
            if chat_stats["total"] > 0:
                print(f"   Avg turns:  {chat_stats['avg_turns']:.1f}")

            print("\n📌 Data Donations (Interesting Papers)")
            print(f"   Total:          {val_stats['total']}")
            print(f"   Unique papers:  {val_stats['unique_papers']}")
            if val_stats["total"] > 0:
                print(f"   Avg priority:   {val_stats['avg_priority']:.1f}")
                dist = val_stats["priority_distribution"]
                if dist:
                    print("   Priority breakdown:")
                    for priority in sorted(dist.keys()):
                        print(f"      {priority}: {dist[priority]}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def feedback_list_command(args: argparse.Namespace) -> int:
    """
    List individual user feedback entries.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat', 'data', or 'all'
        - rating: optional filter for chat donations ('up' or 'down')
        - limit: maximum number of entries to show
        - offset: number of entries to skip

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "all")
            rating_filter = getattr(args, "rating", None)
            limit = getattr(args, "limit", 20)
            offset = getattr(args, "offset", 0)

            if fb_type in ("chat", "all"):
                donations = db.get_chat_donations(limit=limit, rating=rating_filter, offset=offset)
                if donations:
                    print(f"\n💬 Chat Donations ({len(donations)} shown):")
                    print("-" * 60)
                    for entry in donations:
                        ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                        n_turns = len(entry["transcript"])
                        icon = "👍" if entry["rating"] == "up" else "👎"
                        print(f"  ID {entry['id']:5d}  {icon} {entry['rating']:<4}  {n_turns} turns  {ts}")
                else:
                    print("\n💬 No chat donations found.")

            if fb_type in ("data", "all"):
                val_data = db.get_validation_data(limit=limit, offset=offset)
                if val_data:
                    print(f"\n📌 Data Donations ({len(val_data)} shown):")
                    print("-" * 60)
                    for entry in val_data:
                        ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                        term = entry["search_term"] or ""
                        print(
                            f"  ID {entry['id']:5d}  UID {entry['paper_uid']}  "
                            f"priority {entry['priority']}  {ts}  {term}"
                        )
                else:
                    print("\n📌 No data donations found.")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def feedback_browse_command(args: argparse.Namespace) -> int:
    """
    Interactively browse individual user feedback entries.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat' or 'data'
        - rating: optional filter for chat donations ('up' or 'down')
        - sample: optional number of entries to randomly sample

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "chat")
            rating_filter = getattr(args, "rating", None)
            sample_size = getattr(args, "sample", None)

            if fb_type == "chat":
                entries = db.get_chat_donations(rating=rating_filter)
                if not entries:
                    print("No chat donations found.")
                    return 0

                if sample_size and sample_size < len(entries):
                    import random

                    entries = random.sample(entries, sample_size)

                print(f"\n💬 Browsing {len(entries)} chat donation(s)")
                print("Press [Enter] to continue, [q] to quit.\n")

                for i, entry in enumerate(entries, 1):
                    ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                    icon = "👍" if entry["rating"] == "up" else "👎"
                    print(f"--- Entry {i}/{len(entries)} (ID: {entry['id']}) {icon} {entry['rating']} — {ts} ---")
                    for msg in entry["transcript"]:
                        role = msg.get("role", "?")
                        text = msg.get("text", "")
                        print(f"  [{role}] {text}")
                    print()
                    try:
                        choice = input("[Enter] next, [q] quit: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Stopped.")
                        break
                    if choice == "q":
                        print("👋 Stopped.")
                        break

            elif fb_type == "data":
                entries = db.get_validation_data()
                if not entries:
                    print("No data donations found.")
                    return 0

                if sample_size and sample_size < len(entries):
                    import random

                    entries = random.sample(entries, sample_size)

                print(f"\n📌 Browsing {len(entries)} data donation(s)")
                print("Press [Enter] to continue, [q] to quit.\n")

                for i, entry in enumerate(entries, 1):
                    ts = entry["donated_at"].strftime("%Y-%m-%d %H:%M") if entry["donated_at"] else "unknown"
                    term = entry["search_term"] or "(no search term)"
                    print(
                        f"--- Entry {i}/{len(entries)} (ID: {entry['id']}) ---\n"
                        f"  Paper UID:   {entry['paper_uid']}\n"
                        f"  Priority:    {entry['priority']}\n"
                        f"  Search term: {term}\n"
                        f"  Donated at:  {ts}\n"
                    )
                    try:
                        choice = input("[Enter] next, [q] quit: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\n👋 Stopped.")
                        break
                    if choice == "q":
                        print("👋 Stopped.")
                        break
            else:
                print(f"❌ Unknown type '{fb_type}'. Use 'chat' or 'data'.", file=sys.stderr)
                return 1

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def feedback_clear_command(args: argparse.Namespace) -> int:
    """
    Delete user feedback data (chat donations and/or data donations).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - type: 'chat', 'data', or 'all'
        - yes: skip the confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()

            fb_type = getattr(args, "type", "all")

            chat_stats = db.get_chat_donation_stats() if fb_type in ("chat", "all") else {"total": 0}
            val_stats = db.get_validation_data_stats() if fb_type in ("data", "all") else {"total": 0}

            total = chat_stats["total"] + val_stats["total"]
            if total == 0:
                print("No feedback data found — nothing to clear.")
                return 0

            parts = []
            if chat_stats["total"] > 0:
                parts.append(f"{chat_stats['total']} chat donation(s)")
            if val_stats["total"] > 0:
                parts.append(f"{val_stats['total']} data donation(s)")

            print(f"⚠️  This will permanently delete {' and '.join(parts)}.")
            if not getattr(args, "yes", False):
                try:
                    answer = input("Are you sure? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    return 0
                if answer not in ("y", "yes"):
                    print("Aborted.")
                    return 0

            deleted_chat = 0
            deleted_val = 0
            if fb_type in ("chat", "all"):
                deleted_chat = db.delete_chat_donations()
            if fb_type in ("data", "all"):
                deleted_val = db.delete_validation_data()

            if deleted_chat:
                print(f"✅ Deleted {deleted_chat} chat donation(s).")
            if deleted_val:
                print(f"✅ Deleted {deleted_val} data donation(s).")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def registry_upload_command(args: argparse.Namespace) -> int:
    """
    Upload data for a conference (and optionally a specific year) to an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - conference: Conference name
        - year: Conference year (optional; all years if omitted)
        - tag: Optional custom tag

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    config = get_config()
    repository = args.repository or config.registry_repository
    token = args.token or config.github_token
    # Resolve conference name to canonical form; treat None/"all" as "all"
    raw_conf = getattr(args, "conference", None)
    if raw_conf and raw_conf.lower() != "all":
        conference = _resolve_conference_arg(raw_conf) or raw_conf
    else:
        conference = "all"

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    if not token:
        print(
            "❌ Authentication token not specified. Use --token or set GITHUB_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    print("Abstracts Explorer - Registry Upload")
    print("=" * 70)
    print(f"Repository:     {repository}")
    print(f"Conference:     {conference}")
    print(f"Year:           {args.year if args.year else 'all'}")
    if args.tag:
        print(f"Tag:            {args.tag}")
    print("=" * 70)

    try:
        client = RegistryClient(repository=repository, token=token)

        if conference == "all":
            summaries = client.upload_all(
                progress_callback=lambda msg: print(f"  {msg}"),
            )
            print(f"\n✅ Upload complete! Uploaded {len(summaries)} conference(s).")
            for s in summaries:
                print(
                    f"  📦 {s.get('conference', '')}: {s.get('paper_count', 0)} papers, "
                    f"{s.get('embedding_count', 0)} embeddings, "
                    f"{s.get('clustering_cache_count', 0)} cache entries (tag: {s.get('tag', '')})"
                )
        else:
            summary = client.upload(
                conference=conference,
                year=args.year,
                tag=args.tag,
                progress_callback=lambda msg: print(f"  {msg}"),
            )
            print("\n✅ Upload complete!")
            print(f"  📄 Papers:     {summary.get('paper_count', 0)}")
            print(f"  🧮 Embeddings: {summary.get('embedding_count', 0)}")
            print(f"  📦 Cache:      {summary.get('clustering_cache_count', 0)}")
            print(f"  📅 Years:      {summary.get('years', [])}")
            print(f"  🏷️  Tag:        {summary.get('tag', '')}")

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        logger.exception("Upload failed")
        return 1


def registry_download_command(args: argparse.Namespace) -> int:
    """
    Download data for a conference (and optionally a specific year) from an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - conference: Conference name
        - year: Conference year (optional; all years if omitted)
        - tag: Optional custom tag
        - yes: Skip confirmation prompt
        - ignore_embedding_model_mismatch: Proceed despite embedding model mismatch

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import (
        EmbeddingModelMismatchError,
        RegistryClient,
        RegistryError,
    )

    config = get_config()
    repository = args.repository or config.registry_repository
    token = args.token or config.github_token
    # Resolve conference name to canonical form; treat None/"all" as "all"
    raw_conf = getattr(args, "conference", None)
    if raw_conf and raw_conf.lower() != "all":
        conference = _resolve_conference_arg(raw_conf) or raw_conf
    else:
        conference = "all"

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    year_display = str(args.year) if args.year else "all"
    print("Abstracts Explorer - Registry Download")
    print("=" * 70)
    print(f"Repository:     {repository}")
    print(f"Conference:     {conference}")
    print(f"Year:           {year_display}")
    if args.tag:
        print(f"Tag:            {args.tag}")
    print("=" * 70)

    if not args.yes:
        scope = f"{conference}/{year_display}"
        if conference == "all":
            scope = "all conferences"
        print(f"\n⚠️  Warning: This will replace existing data for {scope}!")
        try:
            confirm = input("Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if confirm != "y":
            print("Aborted.")
            return 1

    try:
        client = RegistryClient(repository=repository, token=token)

        embedding_model = getattr(args, "embedding_model", None) or config.embedding_model
        ignore_embedding_model_mismatch = getattr(args, "ignore_embedding_model_mismatch", False)

        try:
            if conference == "all":
                summaries = client.download_all(
                    progress_callback=lambda msg: print(f"  {msg}"),
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                )

                print(f"\n✅ Download complete! Downloaded {len(summaries)} artifact(s).")
                for s in summaries:
                    print(
                        f"  📦 {s.get('conference', '')}: {s.get('paper_count', 0)} papers, "
                        f"{s.get('embedding_count', 0)} embeddings, "
                        f"{s.get('clustering_cache_count', 0)} cache entries"
                    )
            else:
                summary = client.download(
                    conference=conference,
                    year=args.year,
                    tag=args.tag,
                    embedding_model=embedding_model,
                    progress_callback=lambda msg: print(f"  {msg}"),
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                )

                print("\n✅ Download complete!")
                print(f"  📄 Papers:     {summary.get('paper_count', 0)}")
                print(f"  🧮 Embeddings: {summary.get('embedding_count', 0)}")
                print(f"  📦 Cache:      {summary.get('clustering_cache_count', 0)}")
                print(f"  📅 Years:      {summary.get('years', [])}")

                metadata = summary.get("metadata", {})
                if metadata:
                    print(f"\n  ℹ️  Artifact version: {metadata.get('version', 'unknown')}")

        except EmbeddingModelMismatchError as mismatch:
            print(
                f"\n❌ Embedding model mismatch:\n"
                f"  Configured model: '{embedding_model}'\n"
                f"  Artifact model:   '{mismatch.remote_model}'\n"
                f"\nIf both names refer to the same model on different backends,\n"
                f"you can use --ignore-embedding-model-mismatch to proceed.\n"
                f"⚠️  Only use this option if you are certain the models are identical!",
                file=sys.stderr,
            )
            return 1

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        logger.exception("Download failed")
        return 1


def registry_list_command(args: argparse.Namespace) -> int:
    """
    List available tags and artifact info in the registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - tag: Optional specific tag to inspect

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    config = get_config()
    repository = args.repository or config.registry_repository
    token = args.token or config.github_token

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    try:
        client = RegistryClient(repository=repository, token=token)

        if args.tag:
            # Show details for a specific tag
            info = client.get_artifact_info(args.tag)
            print(f"\nArtifact: {repository}:{args.tag}")
            print("=" * 70)

            annotations = info.get("annotations", {})
            if annotations:
                version = annotations.get("com.abstracts-explorer.version", "unknown")
                conference = annotations.get("com.abstracts-explorer.conference", "unknown")
                years = annotations.get("com.abstracts-explorer.years", "unknown")
                papers = annotations.get("com.abstracts-explorer.paper-count", "?")
                embeddings = annotations.get("com.abstracts-explorer.embedding-count", "?")
                emb_model = annotations.get("com.abstracts-explorer.embedding-model", "")
                print(f"  Version:    {version}")
                print(f"  Conference: {conference}")
                print(f"  Years:      {years}")
                if emb_model:
                    print(f"  Model:      {emb_model}")
                print(f"  📄 Papers:     {papers}")
                print(f"  🧮 Embeddings: {embeddings}")

            for layer in info.get("layers", []):
                size = layer.get("size", 0)
                title = layer.get("annotations", {}).get("org.opencontainers.image.title", "")
                if title:
                    print(f"  📦 {title} ({size / 1024 / 1024:.1f} MB)")
        else:
            # List all tags
            tags = client.list_tags()
            print(f"\nAvailable tags in {repository}:")
            print("=" * 70)
            if tags:
                for tag in sorted(tags):
                    print(f"  • {tag}")
            else:
                print("  (no tags found)")

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        logger.exception("List failed")
        return 1


def registry_delete_command(args: argparse.Namespace) -> int:
    """
    Delete registry package versions whose tag version is older than a given version.

    Uses the GitHub Packages API to list all versions of the container package
    and deletes those whose OCI tag version is strictly below *below_version*.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - below_version: Threshold version (e.g. "0.4.0"); versions older than this are deleted
        - conference: Optional conference filter (only tags for this conference are checked)
        - dry_run: When True, print what would be deleted without deleting
        - yes: Skip confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    config = get_config()
    repository = args.repository or config.registry_repository
    token = args.token or config.github_token

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    if not token:
        print(
            "❌ Authentication token not specified. Use --token or set GITHUB_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    below_version = args.below_version
    conference = getattr(args, "conference", None)
    dry_run: bool = getattr(args, "dry_run", False)

    print("Abstracts Explorer - Registry Delete Old Versions")
    print("=" * 70)
    print(f"Repository:     {repository}")
    print(f"Delete below:   {below_version}")
    print(f"Conference:     {conference or 'all'}")
    print(f"Dry-run:        {dry_run}")
    print("=" * 70)

    if not dry_run and not getattr(args, "yes", False):
        confirm = input(
            f"\n⚠️  This will permanently delete all versions below {below_version} from {repository}.\n"
            "Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    try:
        client = RegistryClient(repository=repository, token=token)
        deleted = client.delete_old_versions(
            below_version=below_version,
            conference=conference,
            dry_run=dry_run,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        action = "would be deleted" if dry_run else "deleted"
        print(f"\n{'✅' if not dry_run else '🔍'} Done. {len(deleted)} version(s) {action}.")
        if deleted:
            for entry in deleted:
                tags_str = ", ".join(entry.get("tags", []))
                print(f"  {'[dry-run] ' if dry_run else ''}🗑️  {tags_str}")
        return 0

    except ValueError as e:
        print(f"\n❌ Invalid version: {e}", file=sys.stderr)
        return 1
    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        logger.exception("Delete failed")
        return 1


def pre_process_command(args: argparse.Namespace) -> int:
    """
    Run the full pre-processing pipeline: download → create-embeddings → clustering pre-generate.

    This command chains the three main data-preparation steps in the correct order,
    passing the same ``--conference`` and ``--year`` filters to each step.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - conference: Conference to process (optional; all conferences if omitted)
        - year: Year to process (optional; all available years if omitted)
        - force: Force re-download / reset embeddings / force recompute clustering

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    conference = getattr(args, "conference", None)
    year = getattr(args, "year", None)
    force = getattr(args, "force", False)
    verbose = getattr(args, "verbose", 0)

    print("Abstracts Explorer - Pre-process Pipeline")
    print("=" * 70)
    if conference:
        print(f"Conference: {conference}")
    if year is not None:
        print(f"Year:       {year}")
    print(f"Force:      {force}")
    print("Steps: download → create-embeddings → clustering pre-generate")
    print("=" * 70)

    # Step 1: Download
    print("\n📥 Step 1/3: Downloading conference data...")
    download_args = argparse.Namespace(
        conference=conference,
        year=year,
        output="data/abstracts.json",
        force=force,
        list_plugins=False,
        max_workers=20,
        input_file=None,
        verbose=verbose,
    )
    rc = download_command(download_args)
    if rc != 0:
        print(f"\n❌ Download step failed (exit code {rc})", file=sys.stderr)
        return rc

    # Step 2: Create embeddings
    print("\n🧮 Step 2/3: Creating embeddings...")
    embeddings_args = argparse.Namespace(
        conference=conference,
        year=year,
        collection=config.collection_name,
        lm_studio_url=config.llm_backend_url,
        model=config.embedding_model,
        force=force,
        where=None,
        requests_per_minute=config.requests_per_minute,
        verbose=verbose,
    )
    rc = create_embeddings_command(embeddings_args)
    if rc != 0:
        print(f"\n❌ Create-embeddings step failed (exit code {rc})", file=sys.stderr)
        return rc

    # Step 3: Clustering pre-generate
    print("\n🔮 Step 3/3: Pre-generating clustering results...")
    clustering_args = argparse.Namespace(
        conference=conference,
        year=year,
        collection=config.collection_name,
        force=force,
        requests_per_minute=config.requests_per_minute,
        verbose=verbose,
    )
    rc = pre_generate_clustering_command(clustering_args)
    if rc != 0:
        print(f"\n❌ Clustering pre-generate step failed (exit code {rc})", file=sys.stderr)
        return rc

    print("\n✅ Pre-process pipeline completed successfully!")
    return 0


def main() -> int:
    """
    Main entry point for the CLI.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    # Load config for defaults
    config = get_config()

    parser = argparse.ArgumentParser(
        prog="abstracts-explorer",
        description="Abstracts Explorer - Tools for working with conference abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download NeurIPS 2025 data and create database
  neurips-abstracts download --year 2025
  
  # Generate embeddings for all papers
  neurips-abstracts create-embeddings

  # Search for similar papers
  neurips-abstracts search "graph neural networks for molecular generation"
  
  # Search with custom settings
  neurips-abstracts search "deep learning transformers" \\
    --embeddings-path embeddings/ \\
    --n-results 10 \\
    --show-abstract
  
  # Search with metadata filter
  neurips-abstracts search "reinforcement learning" \\
    --where "decision=Accept (poster)"
        """,
    )

    # Add global verbosity flag
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v for INFO, -vv for DEBUG)",
    )

    # Add version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download conference data and create database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Download papers from various sources using plugins.

Without arguments, downloads ALL conferences for ALL available years
(plugins that require manual input are skipped).

Available plugins:
  neurips  - Official NeurIPS conference data (2020-2025)
  iclr     - Official ICLR conference data (2020-2025)
  icml     - Official ICML conference data (2020-2025)
  ieeevis  - Official IEEE VIS conference data (2025)
  ml4ps    - ML4PS (Machine Learning for Physical Sciences) workshop (2025)
  chi      - ACM CHI conference data (2023-2025, requires manual JSON download)

Examples:
  # Download ALL conference data for ALL available years
  abstracts-explorer download

  # Download all years for a specific conference
  abstracts-explorer download --conference neurips

  # Download a specific conference and year
  abstracts-explorer download --conference neurips --year 2025

  # Load CHI 2024 papers from a pre-downloaded JSON
  abstracts-explorer download --conference chi --year 2024 --input-file chi_2024_program.json

  # List available plugins
  abstracts-explorer download --list-plugins
        """,
    )
    download_parser.add_argument(
        "--output",
        type=str,
        default="data/abstracts.json",
        help="Output path for intermediate JSON file (default: data/abstracts.json)",
    )
    add_conference_year_args(download_parser)
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    download_parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available downloader plugins and exit",
    )
    download_parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum parallel workers for fetching data (default: 20)",
    )
    download_parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        dest="input_file",
        help=(
            "Path to a pre-downloaded conference JSON file. "
            "Required for the 'chi' plugin (download the file from programs.sigchi.org)."
        ),
    )

    # Create embeddings command
    embeddings_parser = subparsers.add_parser(
        "create-embeddings",
        help="Generate embeddings for abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Generate embeddings for paper abstracts using an OpenAI-compatible API.

Without additional arguments, embeds all conference data for all
available years.

With --conference and/or --year, only the matching papers are embedded.

Examples:
  # Embed all papers (default)
  abstracts-explorer create-embeddings

  # Embed only NeurIPS papers
  abstracts-explorer create-embeddings --conference NeurIPS

  # Embed only NeurIPS 2024 papers
  abstracts-explorer create-embeddings --conference NeurIPS --year 2024

  # Embed only papers from 2025 (all conferences)
  abstracts-explorer create-embeddings --year 2025

  # Combine with --where for additional filtering
  abstracts-explorer create-embeddings --conference NeurIPS --where "award IS NOT NULL"
        """,
    )
    add_conference_year_args(embeddings_parser)
    embeddings_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name for the ChromaDB collection (default: {config.collection_name})",
    )
    embeddings_parser.add_argument(
        "--lm-studio-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for LM Studio API (default: {config.llm_backend_url})",
    )
    embeddings_parser.add_argument(
        "--model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    embeddings_parser.add_argument(
        "--force",
        action="store_true",
        help="Reset collection if it already exists",
    )
    embeddings_parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="SQL WHERE clause to filter papers (e.g., \"decision LIKE '%%Accept%%'\")",
    )
    embeddings_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search the vector database for similar papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    search_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    search_parser.add_argument(
        "--n-results",
        type=int,
        default=config.max_context_papers,
        help=f"Number of results to return (default: {config.max_context_papers})",
    )
    search_parser.add_argument(
        "--where",
        type=str,
        default=None,
        help='Metadata filter as key=value pairs, comma-separated (e.g., "decision=Accept (poster)")',
    )
    search_parser.add_argument(
        "--show-abstract",
        action="store_true",
        help="Show paper abstracts in results",
    )
    search_parser.add_argument(
        "--lm-studio-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for LM Studio API (default: {config.llm_backend_url})",
    )
    search_parser.add_argument(
        "--model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    search_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.2,
        dest="distance_threshold",
        help="Maximum L2 distance for a result to be included (default: 1.2)",
    )
    add_conference_year_args(search_parser)

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive RAG chat with papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Start an interactive chat session using RAG to answer questions about papers.",
    )
    chat_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    chat_parser.add_argument(
        "--lm-studio-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for LM Studio API (default: {config.llm_backend_url})",
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default=config.chat_model,
        help=f"Name of the language model for chat (default: {config.chat_model})",
    )
    chat_parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    chat_parser.add_argument(
        "--max-context",
        type=int,
        default=config.max_context_papers,
        help=f"Maximum number of papers to use as context (default: {config.max_context_papers})",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=config.chat_temperature,
        help=f"Sampling temperature for generation (default: {config.chat_temperature})",
    )
    chat_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source papers for each response",
    )
    chat_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export conversation to JSON file",
    )
    add_conference_year_args(chat_parser)

    # Web UI command
    web_parser = subparsers.add_parser(
        "web-ui",
        help="Start the web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Start a Flask web server with a modern UI for exploring papers.",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    web_parser.add_argument(
        "--dev",
        action="store_true",
        help="Use Flask development server instead of production server (Waitress). Note: Use -vv to enable debug mode with any server.",
    )
    web_parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Number of Waitress worker threads (default: 6). Must be >= 1. Ignored when --dev is set.",
    )

    # -------------------------------------------------------------------------
    # Clustering command (parent with sub-subcommands)
    # -------------------------------------------------------------------------
    clustering_parser = subparsers.add_parser(
        "clustering",
        help="Clustering-related commands (run, clear-cache, pre-generate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Clustering-related commands.

Uses agglomerative clustering (ward linkage, distance threshold 150)
with t-SNE dimensionality reduction.

Sub-commands:
  run            Cluster embeddings and export results
  clear-cache    Clear clustering cache from the database
  pre-generate   Pre-generate clustering results for the web UI

Examples:
  # Cluster with default settings
  abstracts-explorer clustering run

  # Clear the cache
  abstracts-explorer clustering clear-cache

  # Pre-generate for all conference/year combinations (default)
  abstracts-explorer clustering pre-generate

  # Pre-generate for a specific conference
  abstracts-explorer clustering pre-generate --conference NeurIPS
        """,
    )
    clustering_subparsers = clustering_parser.add_subparsers(
        dest="clustering_command", help="Clustering sub-commands"
    )

    # clustering run
    cluster_parser = clustering_subparsers.add_parser(
        "run",
        help="Cluster embeddings for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Perform agglomerative clustering with t-SNE dimensionality reduction on paper embeddings.",
    )
    cluster_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    cluster_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export clustering results as JSON (optional)",
    )
    cluster_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of embeddings to process (optional)",
    )
    add_conference_year_args(cluster_parser)

    # clustering clear-cache
    clear_cache_parser = clustering_subparsers.add_parser(
        "clear-cache",
        help="Clear clustering cache from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Clear clustering cache from the database.

This is useful when embeddings change or cache becomes stale.
You can optionally filter by embedding model to clear only specific entries.

Examples:
  # Clear all clustering cache entries
  abstracts-explorer clustering clear-cache

  # Clear cache for a specific embedding model
  abstracts-explorer clustering clear-cache --embedding-model text-embedding-3-large
        """,
    )
    clear_cache_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Only clear cache for this embedding model (optional)",
    )
    add_conference_year_args(clear_cache_parser)

    # clustering pre-generate
    pre_gen_parser = clustering_subparsers.add_parser(
        "pre-generate",
        help="Pre-generate clustering results for the web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Pre-generate clustering results using agglomerative clustering
(ward linkage, distance threshold 150) with t-SNE dimensionality
reduction.  Results are persisted to the database cache so that
the web UI serves them instantly.

Without any arguments, generates clustering for every conference in
the database combined with each individual year and with all years.

With --conference only, generates for that conference with all years
combined AND each individual year.

With --conference and --year, generates for that specific conference
and year only.

Examples:
  # Pre-generate for all conference/year combinations (default)
  abstracts-explorer clustering pre-generate

  # Pre-generate for a specific conference (all years + each year)
  abstracts-explorer clustering pre-generate --conference NeurIPS

  # Pre-generate for a specific conference and year
  abstracts-explorer clustering pre-generate --conference NeurIPS --year 2024

  # Force recompute
  abstracts-explorer clustering pre-generate --force
        """,
    )
    pre_gen_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    add_conference_year_args(pre_gen_parser)
    pre_gen_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recompute even if cache already exists",
    )
    pre_gen_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )

    # MCP Server command
    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Start MCP server for cluster analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a Model Context Protocol (MCP) server for cluster analysis.

The MCP server exposes tools that allow LLM-based assistants to:
- Get most frequently mentioned topics from clusters
- Analyze how topics evolved over years for conferences
- Find recent developments in specific topics
- Generate cluster visualizations

Examples:
  # Start MCP server on default port (8000)
  abstracts-explorer mcp-server
  
  # Start on custom host and port
  abstracts-explorer mcp-server --host 0.0.0.0 --port 8080
  
  # Use stdio transport (for local CLI integration)
  abstracts-explorer mcp-server --transport stdio
        """,
    )
    mcp_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    mcp_parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="sse",
        help="Transport method: sse (HTTP/SSE) or stdio (default: sse)",
    )

    # Eval command (with sub-subcommands)
    eval_parser = subparsers.add_parser(
        "eval",
        help="Automatic evaluation of the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Automatic evaluation of the RAG system.

Sub-commands:
  generate   Generate evaluation Q/A pairs using an LLM
  verify     Interactively verify/edit/delete Q/A pairs
  run        Run evaluation on stored Q/A pairs
  results    Browse and display evaluation results
  clear      Delete all accepted Q/A pairs

Examples:
  # Generate Q/A pairs for all MCP tools
  abstracts-explorer eval generate

  # Generate 5 pairs per tool with 2 follow-ups each
  abstracts-explorer eval generate --n-pairs 5 --n-followups 2

  # Generate pairs only for search_papers tool
  abstracts-explorer eval generate --tools search_papers

  # Verify a random sample of 10 pairs
  abstracts-explorer eval verify --sample 10

  # Run evaluation
  abstracts-explorer eval run

  # Show latest results with details
  abstracts-explorer eval results --detail

  # Show a random sample of 5 detailed results
  abstracts-explorer eval results --detail --sample 5

  # Clear all accepted pairs (with confirmation)
  abstracts-explorer eval clear

  # Clear all accepted pairs without confirmation prompt
  abstracts-explorer eval clear --yes
        """,
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", help="Evaluation sub-commands")

    # eval generate
    eval_gen_parser = eval_subparsers.add_parser(
        "generate",
        help="Generate evaluation Q/A pairs using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_gen_parser.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Number of Q/A pairs to generate per tool (default: 2)",
    )
    eval_gen_parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=None,
        help="Specific MCP tool(s) to generate pairs for (default: all)",
    )
    eval_gen_parser.add_argument(
        "--no-followups",
        action="store_true",
        help="Disable follow-up question generation",
    )
    eval_gen_parser.add_argument(
        "--n-followups",
        type=int,
        default=1,
        help="Number of follow-up turns per initial pair (default: 1)",
    )
    eval_gen_parser.add_argument(
        "--model",
        type=str,
        default=config.chat_model,
        help=f"Chat model for generation (default: {config.chat_model})",
    )
    eval_gen_parser.add_argument(
        "--lm-studio-url",
        type=str,
        default=config.llm_backend_url,
        help=f"LLM backend URL (default: {config.llm_backend_url})",
    )
    add_conference_year_args(eval_gen_parser)

    # eval verify
    eval_verify_parser = eval_subparsers.add_parser(
        "verify",
        help="Interactively verify, edit, or delete Q/A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_verify_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of pairs to randomly sample for review",
    )
    eval_verify_parser.add_argument(
        "--all",
        action="store_true",
        help="Review all pairs (including already verified)",
    )
    add_conference_year_args(eval_verify_parser)

    # eval run
    eval_run_parser = eval_subparsers.add_parser(
        "run",
        help="Run evaluation on stored Q/A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_run_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"ChromaDB collection (default: {config.collection_name})",
    )
    eval_run_parser.add_argument(
        "--model",
        type=str,
        default=config.chat_model,
        help=f"Chat model (default: {config.chat_model})",
    )
    eval_run_parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.embedding_model,
        help=f"Embedding model (default: {config.embedding_model})",
    )
    eval_run_parser.add_argument(
        "--lm-studio-url",
        type=str,
        default=config.llm_backend_url,
        help=f"LLM backend URL (default: {config.llm_backend_url})",
    )
    eval_run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate",
    )
    eval_run_parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified Q/A pairs in evaluation",
    )
    add_conference_year_args(eval_run_parser)

    # eval results
    eval_results_parser = eval_subparsers.add_parser(
        "results",
        help="Browse evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_results_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to display (default: latest)",
    )
    eval_results_parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed per-pair results",
    )
    eval_results_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N results for browsing",
    )
    eval_results_parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all stored evaluation results for the selected run",
    )
    eval_results_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --clear",
    )
    add_conference_year_args(eval_results_parser)

    # eval clear
    eval_clear_parser = eval_subparsers.add_parser(
        "clear",
        help="Delete all accepted (verified) Q/A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_clear_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    add_conference_year_args(eval_clear_parser)

    # Registry command (with sub-subcommands)
    registry_parser = subparsers.add_parser(
        "registry",
        help="Upload/download data to/from OCI-compatible container registries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Upload and download paper databases, embeddings, and clustering cache
to/from OCI-compatible container registries like GitHub Container Registry.

Data is always uploaded/downloaded as a complete unit per conference and year
to prevent inconsistent data between instances.  When --year is omitted,
all available years for the conference are uploaded/downloaded together
with each year stored as its own pair of OCI layers.

Use --conference all to upload/download all available conferences at once.

Sub-commands:
  upload     Upload data for a conference (and optionally a year) to registry
  download   Download data for a conference (and optionally a year) from registry
  list       List available tags in registry

Examples:
  # Upload NeurIPS 2024 data to GitHub Container Registry
  abstracts-explorer registry upload --conference neurips --year 2024 \\
    --repository ghcr.io/thawn/abstracts-data

  # Upload all NeurIPS years
  abstracts-explorer registry upload --conference neurips \\
    --repository ghcr.io/thawn/abstracts-data

  # Download NeurIPS 2024 data from registry
  abstracts-explorer registry download --conference neurips --year 2024 \\
    --repository ghcr.io/thawn/abstracts-data

  # List available tags
  abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data

  # Inspect a specific tag
  abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data \\
    --tag neurips-2024
        """,
    )
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", help="Registry sub-commands")

    # Common registry arguments
    def _add_registry_args(sub_parser: argparse.ArgumentParser) -> None:
        sub_parser.add_argument(
            "--repository",
            "-r",
            type=str,
            default=None,
            help="OCI repository (e.g., ghcr.io/thawn/abstracts-data). "
            "Can also be set via REGISTRY_REPOSITORY env var.",
        )
        sub_parser.add_argument(
            "--token",
            type=str,
            default=None,
            help="Authentication token (e.g., GitHub PAT). Can also be set via GITHUB_TOKEN env var.",
        )

    # registry upload
    registry_upload_parser = registry_subparsers.add_parser(
        "upload",
        help="Upload data for a conference (and optionally a specific year) to registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_registry_args(registry_upload_parser)
    registry_upload_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Conference name (e.g., neurips, iclr) or 'all' for all conferences (default: all).",
    )
    registry_upload_parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=None,
        help="Conference year (e.g., 2024). If omitted, all available years are uploaded.",
    )
    registry_upload_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag (default: derived from conference, year and embedding model)",
    )

    # registry download
    registry_download_parser = registry_subparsers.add_parser(
        "download",
        help="Download data for a conference (and optionally a specific year) from registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_registry_args(registry_download_parser)
    registry_download_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Conference name (e.g., neurips, iclr) or 'all' for all available tags (default: all).",
    )
    registry_download_parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=None,
        help="Conference year (e.g., 2024). If omitted, all years in the artifact are downloaded.",
    )
    registry_download_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag (default: derived from conference, year and embedding model)",
    )
    registry_download_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name for tag derivation. "
        "If omitted, read from local database metadata or EMBEDDING_MODEL env var.",
    )
    registry_download_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    registry_download_parser.add_argument(
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Proceed even if the downloaded artifact uses a different embedding model than configured. "
        "Only use this option when the mismatch is caused by the same model having different names on "
        "different backends (e.g. LM Studio vs. Ollama). "
        "The local embedding model metadata will be updated to match the configured model after download.",
    )

    # registry list
    registry_list_parser = registry_subparsers.add_parser(
        "list",
        help="List available tags in registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_registry_args(registry_list_parser)
    registry_list_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Inspect a specific tag (shows details instead of listing all tags)",
    )
    add_conference_year_args(registry_list_parser)

    # registry delete
    registry_delete_parser = registry_subparsers.add_parser(
        "delete",
        help="Delete registry package versions below a given version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Delete registry package versions whose OCI tag version is strictly below a given threshold.

Uses the GitHub Packages API to list all versions of the container package and permanently
deletes those that are outdated.  Untagged (dangling) versions are left untouched.

Examples:
  # Delete all versions below 0.4.0
  abstracts-explorer registry delete --below-version 0.4.0 \\
    --repository ghcr.io/thawn/abstracts-data

  # Preview without actually deleting (dry-run)
  abstracts-explorer registry delete --below-version 0.4.0 --dry-run \\
    --repository ghcr.io/thawn/abstracts-data

  # Delete only NeurIPS versions below 0.4.0
  abstracts-explorer registry delete --below-version 0.4.0 --conference neurips \\
    --repository ghcr.io/thawn/abstracts-data

  # Skip confirmation prompt
  abstracts-explorer registry delete --below-version 0.4.0 --yes \\
    --repository ghcr.io/thawn/abstracts-data
        """,
    )
    _add_registry_args(registry_delete_parser)
    registry_delete_parser.add_argument(
        "--below-version",
        type=str,
        required=True,
        metavar="VERSION",
        help="Delete all versions strictly below this version (e.g., '0.4.0'). Must be a valid PEP 440 version.",
    )
    registry_delete_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Only delete versions whose tags belong to this conference (optional; all conferences if omitted).",
    )
    registry_delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which versions would be deleted without actually deleting them.",
    )
    registry_delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    # Feedback command (with sub-subcommands)
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Explore user feedback (chat donations and data donations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Explore user feedback collected by the web UI.

Two types of feedback are tracked:
  - Chat donations: anonymized chat transcripts with thumbs up/down ratings.
  - Data donations: anonymized interesting-paper ratings (1-5 priority).

Sub-commands:
  stats   Show summary statistics for all feedback types
  list    List individual feedback entries (with optional filters)
  browse  Interactively browse individual feedback entries
  clear   Delete feedback data (with confirmation)

Examples:
  # Show summary statistics
  abstracts-explorer feedback stats

  # List all chat donations
  abstracts-explorer feedback list --type chat

  # List only thumbs-down chat donations
  abstracts-explorer feedback list --type chat --rating down

  # List data donations (up to 50 entries)
  abstracts-explorer feedback list --type data --limit 50

  # Interactively browse chat donations
  abstracts-explorer feedback browse --type chat

  # Browse a random sample of 5 chat donations
  abstracts-explorer feedback browse --type chat --sample 5

  # Delete all feedback data (with confirmation)
  abstracts-explorer feedback clear

  # Delete only chat donations, no prompt
  abstracts-explorer feedback clear --type chat --yes
        """,
    )
    feedback_subparsers = feedback_parser.add_subparsers(dest="feedback_command", help="Feedback sub-commands")

    # feedback stats
    feedback_subparsers.add_parser(
        "stats",
        help="Show summary statistics for all user feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # feedback list
    feedback_list_parser = feedback_subparsers.add_parser(
        "list",
        help="List individual feedback entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    feedback_list_parser.add_argument(
        "--type",
        choices=["chat", "data", "all"],
        default="all",
        help="Type of feedback to list: 'chat', 'data', or 'all' (default: all)",
    )
    feedback_list_parser.add_argument(
        "--rating",
        choices=["up", "down"],
        default=None,
        help="Filter chat donations by rating (only applies when --type is 'chat' or 'all')",
    )
    feedback_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of entries to show per type (default: 20)",
    )
    feedback_list_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of entries to skip (for pagination, default: 0)",
    )

    # feedback browse
    feedback_browse_parser = feedback_subparsers.add_parser(
        "browse",
        help="Interactively browse individual feedback entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    feedback_browse_parser.add_argument(
        "--type",
        choices=["chat", "data"],
        default="chat",
        help="Type of feedback to browse: 'chat' or 'data' (default: chat)",
    )
    feedback_browse_parser.add_argument(
        "--rating",
        choices=["up", "down"],
        default=None,
        help="Filter chat donations by rating (only applies when --type is 'chat')",
    )
    feedback_browse_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N entries to browse",
    )

    # feedback clear
    feedback_clear_parser = feedback_subparsers.add_parser(
        "clear",
        help="Delete feedback data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    feedback_clear_parser.add_argument(
        "--type",
        choices=["chat", "data", "all"],
        default="all",
        help="Type of feedback to delete: 'chat', 'data', or 'all' (default: all)",
    )
    feedback_clear_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt",
    )

    # Delete-data command
    delete_data_parser = subparsers.add_parser(
        "delete-data",
        help="Delete all data for a specific conference/year from all databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Delete all data for a specific conference and year from every database.

Both --conference and --year are required. The following data will be removed:
  - Papers from the paper database
  - Embeddings from ChromaDB
  - Clustering cache entries

The command shows a summary of what will be deleted and requires the user
to type 'yes' to confirm the operation.

Examples:
  # Delete all NeurIPS 2024 data
  abstracts-explorer delete-data --conference neurips --year 2024

  # Delete without interactive confirmation
  abstracts-explorer delete-data --conference iclr --year 2023 --yes
        """,
    )
    delete_data_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        required=True,
        help="Conference to delete (required). Case-insensitive.",
    )
    delete_data_parser.add_argument(
        "--year",
        "-y",
        type=int,
        required=True,
        help="Year of conference/workshop to delete (required).",
    )
    delete_data_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )

    # Pre-process command
    pre_process_parser = subparsers.add_parser(
        "pre-process",
        help="Run the full pre-processing pipeline (download → create-embeddings → clustering pre-generate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Run the full pre-processing pipeline in order:
  1. download          - Download conference data
  2. create-embeddings - Generate embeddings for abstracts
  3. clustering pre-generate - Pre-generate clustering results for the web UI

All three steps receive the same --conference and --year filters.
Other options for each step use their default values (from config).

Examples:
  # Process all conferences and years
  abstracts-explorer pre-process

  # Process only NeurIPS (all years)
  abstracts-explorer pre-process --conference neurips

  # Process NeurIPS 2025 only
  abstracts-explorer pre-process --conference neurips --year 2025

  # Re-process, overwriting existing data
  abstracts-explorer pre-process --conference neurips --force
        """,
    )
    add_conference_year_args(pre_process_parser)
    pre_process_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reset embeddings even if they already exist",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Setup logging based on verbosity
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "download":
        return download_command(args)
    elif args.command == "create-embeddings":
        return create_embeddings_command(args)
    elif args.command == "pre-process":
        return pre_process_command(args)
    elif args.command == "search":
        return search_command(args)
    elif args.command == "chat":
        return chat_command(args)
    elif args.command == "web-ui":
        return web_ui_command(args)
    elif args.command == "clustering":
        if not hasattr(args, "clustering_command") or not args.clustering_command:
            clustering_parser.print_help()
            return 1
        if args.clustering_command == "run":
            return cluster_embeddings_command(args)
        elif args.clustering_command == "clear-cache":
            return clear_clustering_cache_command(args)
        elif args.clustering_command == "pre-generate":
            return pre_generate_clustering_command(args)
        else:
            clustering_parser.print_help()
            return 1
    elif args.command == "mcp-server":
        return mcp_server_command(args)
    elif args.command == "eval":
        if not hasattr(args, "eval_command") or not args.eval_command:
            eval_parser.print_help()
            return 1
        if args.eval_command == "generate":
            return eval_generate_command(args)
        elif args.eval_command == "verify":
            return eval_verify_command(args)
        elif args.eval_command == "run":
            return eval_run_command(args)
        elif args.eval_command == "results":
            return eval_results_command(args)
        elif args.eval_command == "clear":
            return eval_clear_command(args)
        else:
            eval_parser.print_help()
            return 1
    elif args.command == "delete-data":
        return delete_data_command(args)
    elif args.command == "feedback":
        if not hasattr(args, "feedback_command") or not args.feedback_command:
            feedback_parser.print_help()
            return 1
        if args.feedback_command == "stats":
            return feedback_stats_command(args)
        elif args.feedback_command == "list":
            return feedback_list_command(args)
        elif args.feedback_command == "browse":
            return feedback_browse_command(args)
        elif args.feedback_command == "clear":
            return feedback_clear_command(args)
        else:
            feedback_parser.print_help()
            return 1
    elif args.command == "registry":
        if not hasattr(args, "registry_command") or not args.registry_command:
            registry_parser.print_help()
            return 1
        if args.registry_command == "upload":
            return registry_upload_command(args)
        elif args.registry_command == "download":
            return registry_download_command(args)
        elif args.registry_command == "list":
            return registry_list_command(args)
        elif args.registry_command == "delete":
            return registry_delete_command(args)
        else:
            registry_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
