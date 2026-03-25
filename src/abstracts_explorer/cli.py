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
from typing import Optional

import argcomplete

from tqdm import tqdm

from .config import get_config
from .database import DatabaseManager
from .embeddings import EmbeddingsManager, EmbeddingsError
from .clustering import perform_clustering, compute_clusters_with_cache, ClusteringError
from .rag import RAGChat, RAGError
from .plugins import get_plugin, list_plugins, list_plugin_names
from .mcp_server import run_mcp_server
from .evaluation import (
    EvaluationError,
    Evaluator,
    format_eval_summary,
    format_eval_result_detail,
)

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


def create_embeddings_command(args: argparse.Namespace) -> int:
    """
    Create embeddings database for abstracts.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name for the ChromaDB collection
        - lm_studio_url: URL for OpenAI-compatible API
        - model: Name of the embedding model
        - force: Whether to reset existing collection
        - where: SQL WHERE clause to filter papers

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    print("Abstracts Explorer - Embeddings Generator")
    print("=" * 70)
    print(f"Database: {config.database_url}")
    print(f"Embedding DB:   {config.embedding_db}")
    print(f"Collection: {args.collection}")
    print(f"Model:    {args.model}")
    print(f"API URL: {args.lm_studio_url}")
    rate_limit_str = f"{args.requests_per_minute} req/min" if args.requests_per_minute > 0 else "disabled"
    print(f"Rate limit: {rate_limit_str}")
    print("=" * 70)

    # Check paper count
    with DatabaseManager() as db:
        total_papers = db.get_paper_count()
        print(f"\n📊 Found {total_papers:,} papers in database")

        if args.where:
            # Count papers matching filter
            filtered = db.query(f"SELECT COUNT(*) as count FROM papers WHERE {args.where}")
            filtered_count = filtered[0]["count"] if filtered else 0
            print(f"📊 Filter will process {filtered_count:,} papers")

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

        # Create or reset collection
        if args.force:
            print(f"🔄 Resetting existing collection '{args.collection}'...")
        else:
            print(f"📁 Creating collection '{args.collection}'...")

        em.create_collection(reset=args.force)

        # Generate embeddings with progress bar
        print("\n🚀 Generating embeddings...")

        # Determine total count for progress bar
        with DatabaseManager() as db:
            if args.where:
                count_result = db.query(f"SELECT COUNT(*) as count FROM papers WHERE {args.where}")
                total_count = count_result[0]["count"] if count_result else 0
            else:
                total_count = db.get_paper_count()

        # Create progress bar
        with tqdm(total=total_count, desc="Embedding papers", unit="papers") as pbar:

            def update_progress(current: int, total: int) -> None:
                pbar.n = current
                pbar.total = total
                pbar.refresh()

            embedded_count = em.embed_from_database(
                where_clause=args.where,
                progress_callback=update_progress,
                force_recreate=args.force,
            )

        print(f"✅ Successfully generated embeddings for {embedded_count:,} papers")

        # Show collection stats
        stats = em.get_collection_stats()
        print("\n📊 Collection Statistics:")
        print(f"   Name:  {stats['name']}")
        print(f"   Count: {stats['count']:,} documents")

        em.close()

        print(f"\n💾 Vector database saved to: {config.embedding_db}")
        print("\nYou can now use the 'search' command or the search_similar() method to find relevant papers!")

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

    print("NeurIPS Semantic Search")
    print("=" * 70)
    print(f"Query: {args.query}")
    print(f"Embeddings: {config.embedding_db}")
    print(f"Collection: {args.collection}")
    print(f"Results: {args.n_results}")
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

        # Parse metadata filter if provided
        where_filter = None
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

        num_results = len(results["ids"][0])
        print(f"✅ Found {num_results} similar paper(s):\n")

        for i in range(num_results):
            paper_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance if distance <= 1 else 0
            document = results["documents"][0][i] if "documents" in results else ""

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

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    try:
        config = get_config()

        print("=" * 70)
        print("NeurIPS RAG Chat")
        print("=" * 70)
        print(f"Embeddings: {config.embedding_db}")
        print(f"Collection: {args.collection}")
        print(f"Chat Model: {args.model}")
        print(f"Embedding Model: {args.embedding_model}")
        print(f"API URL: {args.lm_studio_url}")
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
                result = chat.query(user_input)
                print("\r" + " " * 50 + "\r", end="")  # Clear the line

                # Display response
                print(f"Assistant (based on {result['metadata']['n_papers']} papers):")
                print(result["response"])
                print()

                # Show source papers if requested
                if args.show_sources and result["papers"]:
                    print("📚 Source papers:")
                    for i, paper in enumerate(result["papers"], 1):
                        print(f"  {i}. {paper['title']} (similarity: {paper['similarity']:.3f})")
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


def download_command(args: argparse.Namespace) -> int:
    """
    Download NeurIPS data and create database.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - plugin: Name of the downloader plugin to use
        - year: Year of NeurIPS conference
        - output: Path for the database file
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
    plugin_name = getattr(args, "plugin", "neurips")

    # Get the plugin
    plugin = get_plugin(plugin_name)
    if not plugin:
        print(f"❌ Error: Plugin '{plugin_name}' not found", file=sys.stderr)
        print(f"\nAvailable plugins: {', '.join(list_plugin_names())}", file=sys.stderr)
        print("\nUse --list-plugins to see details", file=sys.stderr)
        return 1

    print(f"Using plugin: {plugin.plugin_description}")
    print(f"Downloading {plugin.plugin_name}...")
    print("=" * 70)

    try:
        # Prepare kwargs for plugin
        kwargs = {}

        # Add plugin-specific options
        if plugin_name == "ml4ps":
            kwargs["max_workers"] = getattr(args, "max_workers", 20)

        if plugin_name == "chi":
            input_file = getattr(args, "input_file", None)
            if input_file:
                kwargs["input_path"] = input_file

        # Download data using plugin
        json_path = output_path.parent / f"{plugin_name}_{args.year}.json"
        papers = plugin.download(year=args.year, output_path=str(json_path), force_download=args.force, **kwargs)

        print(f"✅ Downloaded {len(papers):,} papers")

        # Create database using config
        print("\n📊 Creating database using configuration")
        with DatabaseManager() as db:
            db.create_tables()
            count = db.add_papers(papers)
            print(f"✅ Loaded {count:,} papers into database")

        config = get_config()
        print(f"\n💾 Database updated: {config.database_url}")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        if args.verbose > 0:
            traceback.print_exc()
        return 1


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

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - reduction_method: Dimensionality reduction method
        - n_components: Number of components for reduction
        - clustering_method: Clustering algorithm to use
        - n_clusters: Number of clusters (for kmeans/agglomerative)
        - eps: DBSCAN eps parameter
        - min_samples: DBSCAN min_samples parameter
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
    print(f"Reduction:   {args.reduction_method} (n_components={args.n_components})")
    print(f"Clustering:  {args.clustering_method}", end="")

    if args.clustering_method.lower() == "kmeans" or args.clustering_method.lower() == "agglomerative":
        print(f" (n_clusters={args.n_clusters})")
    elif args.clustering_method.lower() == "dbscan":
        print(f" (eps={args.eps}, min_samples={args.min_samples})")
    else:
        print()

    if args.limit:
        print(f"Limit:       {args.limit} papers")
    print("=" * 70)

    try:
        # Prepare kwargs for clustering
        kwargs = {}
        if args.clustering_method.lower() == "dbscan":
            kwargs["eps"] = args.eps
            kwargs["min_samples"] = args.min_samples

        # Perform clustering
        print("\n🚀 Starting clustering pipeline...")
        results = perform_clustering(
            collection_name=args.collection,
            reduction_method=args.reduction_method,
            n_components=args.n_components,
            clustering_method=args.clustering_method,
            n_clusters=args.n_clusters,
            output_path=args.output,
            limit=args.limit,
            **kwargs,
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


def pre_generate_clustering_command(args: argparse.Namespace) -> int:
    """
    Pre-generate default clustering results and hierarchical labels.

    Runs agglomerative clustering with default settings and generates
    hierarchical labels, persisting both to the database cache.  This
    command should be run once after ``create-embeddings`` to warm the
    cache so that the first web-UI request for clusters is fast.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the ChromaDB collection
        - linkage: Agglomerative linkage method
        - n_clusters: Number of clusters (0 = auto-calculate via distance_threshold)
        - distance_threshold: Agglomerative distance threshold (overrides n_clusters)
        - reduction_method: Dimensionality reduction method for visualization
        - conference: Single conference to filter by (optional)
        - years: List of years to filter by (optional)

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

    linkage = args.linkage
    # distance_threshold takes precedence over n_clusters.
    # When distance_threshold is set (the default), n_clusters is ignored.
    distance_threshold: Optional[float] = getattr(args, "distance_threshold", None)
    if distance_threshold is not None and distance_threshold <= 0:
        distance_threshold = None
    # args.n_clusters is 0 when the user wants auto-calculation.
    # We pass None to compute_clusters_with_cache so it calculates
    # the default based on the corpus size.
    n_clusters_arg: Optional[int] = args.n_clusters if args.n_clusters > 0 else None
    # If a distance_threshold is supplied, n_clusters is irrelevant (same as web UI).
    if distance_threshold is not None:
        n_clusters_arg = None
    reduction_method = args.reduction_method

    # Build optional conference/year filters
    raw_conference: Optional[str] = getattr(args, "conference", None) or None
    years: Optional[list] = getattr(args, "years", None) or None

    # Resolve conference name case-insensitively against stored names in the DB.
    conferences: Optional[list] = None
    if raw_conference:
        try:
            with DatabaseManager() as _db_resolve:
                opts = _db_resolve.get_filter_options()
                stored_conferences: list = opts.get("conferences", [])
            match = next(
                (c for c in stored_conferences if c.lower() == raw_conference.lower()),
                None,
            )
            if match is None:
                # No exact case-insensitive match; use the raw value and let
                # compute_clusters_with_cache surface any "no papers" error.
                conferences = [raw_conference]
            else:
                if match != raw_conference:
                    print(f"ℹ️  Resolved conference '{raw_conference}' → '{match}'")
                conferences = [match]
        except Exception:
            # Fall back gracefully if the DB isn't available yet.
            conferences = [raw_conference]

    print("Abstracts Explorer - Pre-generate Clustering")
    print("=" * 70)
    print(f"Embeddings:       {config.embedding_db}")
    print(f"Collection:       {args.collection}")
    print(f"Clustering:       agglomerative (linkage={linkage})")
    if distance_threshold is not None:
        print(f"Dist. threshold:  {distance_threshold}")
    else:
        print(f"N-clusters:       {'auto' if n_clusters_arg is None else n_clusters_arg}")
    print(f"Reduction:        {reduction_method} (for initial visualization)")
    if conferences:
        print(f"Conference:       {conferences[0]}")
    if years:
        print(f"Years:            {', '.join(str(y) for y in years)}")
    print("=" * 70)

    try:
        em = EmbeddingsManager(
            collection_name=args.collection,
        )
        em.connect()
        em.create_collection()

        # Build agglomerative kwargs to match web UI defaults exactly.
        clustering_kwargs: dict = {"linkage": linkage}
        if distance_threshold is not None:
            clustering_kwargs["distance_threshold"] = distance_threshold

        with DatabaseManager() as db:
            print("\n🚀 Starting agglomerative clustering pipeline...")
            results = compute_clusters_with_cache(
                embeddings_manager=em,
                database=db,
                embedding_model=config.embedding_model,
                reduction_method=reduction_method,
                n_components=2,
                clustering_method="agglomerative",
                n_clusters=n_clusters_arg,
                limit=None,
                force=args.force,
                conferences=conferences,
                years=years,
                **clustering_kwargs,
            )

        stats = results.get("statistics", {})
        print("\n📊 Clustering Results:")
        print(
            f"   Total papers:  {stats.get('total_papers', 'N/A'):,}"
            if isinstance(stats.get("total_papers"), int)
            else f"   Total papers:  {stats.get('total_papers', 'N/A')}"
        )
        print(f"   Clusters:      {stats.get('n_clusters', 'N/A')}")
        print("\n✅ Default clustering results and hierarchical labels cached successfully.")
        return 0

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
        print("  - get_cluster_topics: Get most frequently mentioned topics from clusters")
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


def registry_upload_command(args: argparse.Namespace) -> int:
    """
    Upload data artifacts to an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - tag: Tag for the upload
        - conference: List of conferences to include
        - paper_db: Upload paper database
        - embedding_db: Upload embedding database
        - all: Upload all databases

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from .registry import RegistryClient, RegistryError

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

    # Determine what to upload
    upload_paper_db = args.paper_db or args.all or (not args.paper_db and not args.embedding_db)
    upload_embedding_db = args.embedding_db or args.all or (not args.paper_db and not args.embedding_db)

    conferences = args.conference if args.conference else None

    print("Abstracts Explorer - Registry Upload")
    print("=" * 70)
    print(f"Repository:     {repository}")
    print(f"Tag:            {args.tag}")
    print(f"Paper DB:       {'Yes' if upload_paper_db else 'No'}")
    print(f"Embedding DB:   {'Yes' if upload_embedding_db else 'No'}")
    if conferences:
        print(f"Conferences:    {', '.join(conferences)}")
    print("=" * 70)

    try:
        client = RegistryClient(repository=repository, token=token)
        summary = client.upload(
            tag=args.tag,
            paper_db=upload_paper_db,
            embedding_db=upload_embedding_db,
            conferences=conferences,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

        print("\n✅ Upload complete!")
        for layer in summary.get("layers", []):
            layer_type = layer.get("type", "unknown")
            if layer_type == "paper-db":
                print(f"  📄 Paper database: {layer.get('papers', 0)} papers")
            elif layer_type == "embedding-db":
                print(f"  🧮 Embedding database: {layer.get('embeddings', 0)} embeddings")

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
    Download data artifacts from an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - tag: Tag to download
        - paper_db: Download paper database
        - embedding_db: Download embedding database
        - all: Download all databases
        - merge: Merge with existing data

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from .registry import RegistryClient, RegistryError

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

    # Determine what to download
    dl_paper_db = args.paper_db or args.all or (not args.paper_db and not args.embedding_db)
    dl_embedding_db = args.embedding_db or args.all or (not args.paper_db and not args.embedding_db)

    print("Abstracts Explorer - Registry Download")
    print("=" * 70)
    print(f"Repository:     {repository}")
    print(f"Tag:            {args.tag}")
    print(f"Paper DB:       {'Yes' if dl_paper_db else 'No'}")
    print(f"Embedding DB:   {'Yes' if dl_embedding_db else 'No'}")
    print(f"Merge:          {'Yes' if args.merge else 'No (replace)'}")
    print("=" * 70)

    if not args.merge:
        print("\n⚠️  Warning: This will replace your existing data!")
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
        summary = client.download(
            tag=args.tag,
            paper_db=dl_paper_db,
            embedding_db=dl_embedding_db,
            merge=args.merge,
            progress_callback=lambda msg: print(f"  {msg}"),
        )

        print("\n✅ Download complete!")
        for layer in summary.get("layers", []):
            layer_type = layer.get("type", "unknown")
            if layer_type == "paper-db":
                print(f"  📄 Paper database: {layer.get('papers', 0)} papers imported")
            elif layer_type == "embedding-db":
                print(f"  🧮 Embedding database: {layer.get('embeddings', 0)} embeddings imported")

        metadata = summary.get("metadata", {})
        if metadata:
            print(f"\n  ℹ️  Artifact version: {metadata.get('version', 'unknown')}")
            if metadata.get("conferences"):
                print(f"  ℹ️  Conferences: {', '.join(metadata['conferences'])}")

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
    from .registry import RegistryClient, RegistryError

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

            metadata = info.get("metadata", {})
            if metadata:
                print(f"  Version:    {metadata.get('version', 'unknown')}")
                print(f"  Created:    {metadata.get('created_at', 'unknown')}")
                if metadata.get("conferences"):
                    print(f"  Conferences: {', '.join(metadata['conferences'])}")
                if metadata.get("embedding_model"):
                    print(f"  Embedding:  {metadata['embedding_model']}")

            for layer in info.get("layers", []):
                media_type = layer.get("media_type", "")
                size = layer.get("size", 0)
                annotations = layer.get("annotations", {})

                if "paper-db" in media_type:
                    count = annotations.get("com.abstracts-explorer.paper-count", "?")
                    print(f"  📄 Paper DB: {count} papers ({size / 1024 / 1024:.1f} MB)")
                elif "embedding-db" in media_type:
                    count = annotations.get("com.abstracts-explorer.embedding-count", "?")
                    print(f"  🧮 Embeddings: {count} embeddings ({size / 1024 / 1024:.1f} MB)")
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

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download NeurIPS data and create database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Download papers from various sources using plugins.

Available plugins:
  neurips  - Official NeurIPS conference data (2013-2025)
  ml4ps    - ML4PS (Machine Learning for Physical Sciences) workshop (2025)
  chi      - ACM CHI conference data (2023-2025, requires manual JSON download)

Examples:
  # Download NeurIPS 2025 papers
  neurips-abstracts download --plugin neurips --year 2025

  # Download ML4PS 2025 workshop papers with abstracts
  neurips-abstracts download --plugin ml4ps --year 2025

  # Load CHI 2024 papers from a pre-downloaded JSON
  abstracts-explorer download --plugin chi --year 2024 --input-file chi_2024_program.json

  # List available plugins
  neurips-abstracts download --list-plugins
        """,
    )
    download_parser.add_argument(
        "--plugin",
        type=str,
        default="neurips",
        help="Downloader plugin to use (default: neurips). Use --list-plugins to see available plugins",
    )
    download_parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year of conference/workshop (default: 2025)",
    )
    download_parser.add_argument(
        "--output",
        type=str,
        default="data/abstracts.json",
        help="Output path for intermediate JSON file (default: data/abstracts.json)",
    )
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
    )
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

Sub-commands:
  run            Cluster embeddings and export results
  clear-cache    Clear clustering cache from the database
  pre-generate   Pre-generate default clustering results and hierarchical labels

Examples:
  # Cluster with default settings
  abstracts-explorer clustering run

  # Clear the cache
  abstracts-explorer clustering clear-cache

  # Warm the cache for ML4PS@NeurIPS
  abstracts-explorer clustering pre-generate --conference "ML4PS@NeurIPS"
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
        description="Perform dimensionality reduction and clustering on paper embeddings.",
    )
    cluster_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    cluster_parser.add_argument(
        "--reduction-method",
        type=str,
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    cluster_parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction (default: 2)",
    )
    cluster_parser.add_argument(
        "--clustering-method",
        type=str,
        choices=["kmeans", "dbscan", "agglomerative"],
        default="kmeans",
        help="Clustering algorithm (default: kmeans)",
    )
    cluster_parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for kmeans/agglomerative (default: 5)",
    )
    cluster_parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter (default: 0.5)",
    )
    cluster_parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter (default: 5)",
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

    # clustering pre-generate
    pre_gen_parser = clustering_subparsers.add_parser(
        "pre-generate",
        help="Pre-generate default clustering results and hierarchical labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Pre-generate default clustering results and hierarchical labels.

Runs agglomerative clustering with the given settings and persists the
results (including hierarchical labels) to the database cache.  Run this
command once after create-embeddings to warm the cache so that the first
web-UI request for clusters is served instantly.

Examples:
  # Pre-generate for ML4PS@NeurIPS (all years)
  abstracts-explorer clustering pre-generate --conference "ML4PS@NeurIPS"

  # Pre-generate for a specific conference and year
  abstracts-explorer clustering pre-generate --conference NeurIPS --years 2024

  # Pre-generate for multiple years
  abstracts-explorer clustering pre-generate --conference NeurIPS --years 2023 2024

  # Use a specific linkage and force recompute
  abstracts-explorer clustering pre-generate --linkage complete --force
        """,
    )
    pre_gen_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    pre_gen_parser.add_argument(
        "--linkage",
        type=str,
        choices=["ward", "complete", "average", "single"],
        default="ward",
        help="Agglomerative linkage method (default: ward)",
    )
    pre_gen_parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of clusters (default: 0 = auto-calculate). Ignored when --distance-threshold is set.",
    )
    pre_gen_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=150.0,
        help=(
            "Agglomerative distance threshold that controls the number of clusters "
            "(default: 150, matching the web UI default). Set to 0 to disable and "
            "use --n-clusters instead."
        ),
    )
    pre_gen_parser.add_argument(
        "--reduction-method",
        type=str,
        choices=["pca", "tsne", "umap"],
        default="tsne",
        help="Dimensionality reduction method for the initial visualization (default: tsne, matching web UI default)",
    )
    pre_gen_parser.add_argument(
        "--conference",
        type=str,
        default=None,
        help="Only cluster papers from this conference (default: all conferences)",
    )
    pre_gen_parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        metavar="YEAR",
        help="Only cluster papers from these year(s), e.g. --years 2024 or --years 2023 2024 (default: all years)",
    )
    pre_gen_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recompute even if cache already exists",
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

    # Registry command (with sub-subcommands)
    registry_parser = subparsers.add_parser(
        "registry",
        help="Upload/download data to/from OCI-compatible container registries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Upload and download paper databases, embeddings, and clustering cache
to/from OCI-compatible container registries like GitHub Container Registry.

Sub-commands:
  upload     Upload databases to registry
  download   Download databases from registry
  list       List available tags in registry

Examples:
  # Upload all data to GitHub Container Registry
  abstracts-explorer registry upload --repository ghcr.io/owner/abstracts-data --token ghp_xxxx

  # Upload only paper database for a specific conference
  abstracts-explorer registry upload --repository ghcr.io/owner/abstracts-data --paper-db --conference neurips

  # Download and merge with existing data
  abstracts-explorer registry download --repository ghcr.io/owner/abstracts-data --merge

  # List available tags
  abstracts-explorer registry list --repository ghcr.io/owner/abstracts-data

  # Inspect a specific tag
  abstracts-explorer registry list --repository ghcr.io/owner/abstracts-data --tag neurips-2024
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
            help="OCI repository (e.g., ghcr.io/owner/abstracts-data). "
            "Can also be set via REGISTRY_REPOSITORY env var.",
        )
        sub_parser.add_argument(
            "--token",
            "-t",
            type=str,
            default=None,
            help="Authentication token (e.g., GitHub PAT). Can also be set via GITHUB_TOKEN env var.",
        )

    # registry upload
    registry_upload_parser = registry_subparsers.add_parser(
        "upload",
        help="Upload databases to registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_registry_args(registry_upload_parser)
    registry_upload_parser.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Tag for the upload (default: latest)",
    )
    registry_upload_parser.add_argument(
        "--conference",
        "-c",
        action="append",
        help="Specific conference(s) to include (can be repeated)",
    )
    registry_upload_parser.add_argument(
        "--paper-db",
        action="store_true",
        help="Upload paper database only",
    )
    registry_upload_parser.add_argument(
        "--embedding-db",
        action="store_true",
        help="Upload embedding database only",
    )
    registry_upload_parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all databases (default behavior when no specific database is selected)",
    )

    # registry download
    registry_download_parser = registry_subparsers.add_parser(
        "download",
        help="Download databases from registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_registry_args(registry_download_parser)
    registry_download_parser.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Tag to download (default: latest)",
    )
    registry_download_parser.add_argument(
        "--paper-db",
        action="store_true",
        help="Download paper database only",
    )
    registry_download_parser.add_argument(
        "--embedding-db",
        action="store_true",
        help="Download embedding database only",
    )
    registry_download_parser.add_argument(
        "--all",
        action="store_true",
        help="Download all databases (default behavior when no specific database is selected)",
    )
    registry_download_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing data instead of replacing",
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
        else:
            registry_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
