"""
Command-line interface for neurips-abstracts package.

This module provides CLI commands for working with NeurIPS abstracts,
including downloading data, creating databases, and generating embeddings.
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .config import get_config
from .database import DatabaseManager
from .downloader import download_neurips_data
from .embeddings import EmbeddingsManager, EmbeddingsError
from .rag import RAGChat, RAGError
from .plugins import get_plugin, list_plugins, list_plugin_names


def setup_logging(verbosity: int) -> None:
    """
    Configure logging based on verbosity level.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_embeddings_command(args: argparse.Namespace) -> int:
    """
    Create embeddings database for NeurIPS abstracts.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - db_path: Path to the SQLite database with papers
        - output: Path for the ChromaDB vector database
        - collection: Name for the ChromaDB collection
        - lm_studio_url: URL for LM Studio API
        - model: Name of the embedding model
        - force: Whether to reset existing collection
        - where: SQL WHERE clause to filter papers

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    db_path = Path(args.db_path)
    output_path = Path(args.output)

    # Validate database exists
    if not db_path.exists():
        print(f"❌ Error: Database file not found: {db_path}", file=sys.stderr)
        print("\nYou can create a database using:", file=sys.stderr)
        print(f"  neurips-abstracts download --output {db_path}", file=sys.stderr)
        return 1

    print("NeurIPS Embeddings Generator")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Output:   {output_path}")
    print(f"Collection: {args.collection}")
    print(f"Model:    {args.model}")
    print(f"LM Studio: {args.lm_studio_url}")
    print("=" * 70)

    # Check paper count
    with DatabaseManager(db_path) as db:
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
            chroma_path=output_path,
            collection_name=args.collection,
        )

        # Test connection
        print("🔌 Testing LM Studio connection...")
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to LM Studio!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - LM Studio is running at {args.lm_studio_url}", file=sys.stderr)
            print(f"  - The {args.model} model is loaded", file=sys.stderr)
            return 1
        print("✅ Successfully connected to LM Studio\n")

        # Connect to ChromaDB
        em.connect()

        # Create or reset collection
        if args.force and output_path.exists():
            print(f"🔄 Resetting existing collection '{args.collection}'...")
        else:
            print(f"📁 Creating collection '{args.collection}'...")

        em.create_collection(reset=args.force)

        # Generate embeddings with progress bar
        print(f"\n🚀 Generating embeddings...")

        # Determine total count for progress bar
        with DatabaseManager(db_path) as db:
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
                db_path=db_path,
                where_clause=args.where,
                progress_callback=update_progress,
            )

        print(f"✅ Successfully generated embeddings for {embedded_count:,} papers")

        # Show collection stats
        stats = em.get_collection_stats()
        print("\n📊 Collection Statistics:")
        print(f"   Name:  {stats['name']}")
        print(f"   Count: {stats['count']:,} documents")

        em.close()

        print(f"\n💾 Vector database saved to: {output_path}")
        print("\nYou can now use the search_similar() method to find relevant papers!")

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
        - embeddings_path: Path to ChromaDB vector database
        - collection: Name of the ChromaDB collection
        - n_results: Number of results to return
        - where: Metadata filter conditions
        - show_abstract: Whether to show paper abstracts
        - lm_studio_url: URL for LM Studio API
        - model: Name of the embedding model

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    embeddings_path = Path(args.embeddings_path)

    # Validate embeddings database exists
    if not embeddings_path.exists():
        print(f"❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
        print("\nYou can create embeddings using:", file=sys.stderr)
        print("  neurips-abstracts create-embeddings --db-path <database.db>", file=sys.stderr)
        return 1

    print("NeurIPS Semantic Search")
    print("=" * 70)
    print(f"Query: {args.query}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Collection: {args.collection}")
    print(f"Results: {args.n_results}")
    print("=" * 70)

    try:
        # Initialize embeddings manager
        em = EmbeddingsManager(
            lm_studio_url=args.lm_studio_url,
            model_name=args.model,
            chroma_path=embeddings_path,
            collection_name=args.collection,
        )

        # Test LM Studio connection
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to LM Studio!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - LM Studio is running at {args.lm_studio_url}", file=sys.stderr)
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

        # Try to open database to get author names
        db_manager = None
        if args.db_path:
            db_path = Path(args.db_path)
            if db_path.exists():
                try:
                    db_manager = DatabaseManager(db_path)
                    db_manager.connect()
                except Exception as e:
                    print(f"⚠️  Could not open database for author names: {e}", file=sys.stderr)

        for i in range(num_results):
            paper_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance if distance <= 1 else 0
            document = results["documents"][0][i] if "documents" in results else ""

            print(f"{i + 1}. Paper ID: {paper_id}")
            print(f"   Title: {metadata.get('title', 'N/A')}")

            # Get author names from database if available
            authors_display = metadata.get("authors", "N/A")
            if db_manager:
                try:
                    authors = db_manager.get_paper_authors(int(paper_id))
                    if authors:
                        author_names = [author["fullname"] for author in authors]
                        authors_display = ", ".join(author_names)
                except Exception:
                    pass  # Fall back to author IDs

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

        if db_manager:
            db_manager.close()

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
    Interactive RAG chat with NeurIPS papers.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - embeddings_path: Path to ChromaDB database
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
        embeddings_path = Path(args.embeddings_path)

        print("=" * 70)
        print("NeurIPS RAG Chat")
        print("=" * 70)
        print(f"Embeddings: {embeddings_path}")
        print(f"Collection: {args.collection}")
        print(f"Chat Model: {args.model}")
        print(f"Embedding Model: {args.embedding_model}")
        print(f"LM Studio: {args.lm_studio_url}")
        print("=" * 70)

        # Check embeddings exist
        if not embeddings_path.exists():
            print(f"\n❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
            print("\nYou can create embeddings using:", file=sys.stderr)
            print("  neurips-abstracts create-embeddings --db-path <database.db>", file=sys.stderr)
            return 1

        # Initialize embeddings manager
        em = EmbeddingsManager(
            chroma_path=embeddings_path,
            collection_name=args.collection,
            lm_studio_url=args.lm_studio_url,
            model_name=args.embedding_model,
        )

        # Test LM Studio connection
        print("\n🔌 Testing LM Studio connection...")
        if not em.test_lm_studio_connection():
            print("\n❌ Failed to connect to LM Studio!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - LM Studio is running at {args.lm_studio_url}", file=sys.stderr)
            print("  - A language model is loaded", file=sys.stderr)
            return 1
        print("✅ Successfully connected to LM Studio")

        # Connect to embeddings
        em.connect()

        # Get or create the collection (should already exist for chat)
        em.create_collection(reset=False)

        # Get collection stats
        stats = em.get_collection_stats()
        print(f"\n📊 Loaded {stats['count']:,} papers from collection '{stats['name']}'")

        # Initialize database connection
        from neurips_abstracts.database import DatabaseManager
        from neurips_abstracts.config import get_config

        config_obj = get_config()
        db = DatabaseManager(config_obj.paper_db_path)
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
    import neurips_abstracts.plugins.neurips_downloader
    import neurips_abstracts.plugins.ml4ps_downloader

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

        # Download data using plugin
        json_path = output_path.parent / f"{plugin_name}_{args.year}.json"
        papers = plugin.download(year=args.year, output_path=str(json_path), force_download=args.force, **kwargs)

        print(f"✅ Downloaded {len(papers):,} papers")

        # Create database
        print(f"\n📊 Creating database: {output_path}")
        with DatabaseManager(output_path) as db:
            db.create_tables()
            count = db.add_papers(papers)
            print(f"✅ Loaded {count:,} papers into database")

        print(f"\n💾 Database saved to: {output_path}")
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
        - debug: Enable debug mode

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Try to import Flask - if it fails, give helpful error
        try:
            from neurips_abstracts.web_ui import run_server
        except ImportError:
            print("\n❌ Web UI dependencies not installed!", file=sys.stderr)
            print("\nThe web UI requires Flask. Install it with:", file=sys.stderr)
            print("  pip install neurips-abstracts[web]", file=sys.stderr)
            print("\nOr install Flask manually:", file=sys.stderr)
            print("  pip install flask flask-cors", file=sys.stderr)
            return 1

        # Start the server
        run_server(host=args.host, port=args.port, debug=args.debug)
        return 0

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
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
        prog="neurips-abstracts",
        description="Tools for working with NeurIPS conference abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download NeurIPS 2025 data and create database
  neurips-abstracts download --year 2025 --output data/neurips_2025.db
  
  # Generate embeddings for all papers
  neurips-abstracts create-embeddings --db-path data/neurips_2025.db
  
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

Examples:
  # Download NeurIPS 2025 papers
  neurips-abstracts download --plugin neurips --year 2025 --output data/neurips_2025.db
  
  # Download ML4PS 2025 workshop papers with abstracts
  neurips-abstracts download --plugin ml4ps --year 2025 --output data/ml4ps_2025.db
  
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
        default=config.paper_db_path,
        help=f"Output database file path (default: {config.paper_db_path})",
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

    # Create embeddings command
    embeddings_parser = subparsers.add_parser(
        "create-embeddings",
        help="Generate embeddings for NeurIPS abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    embeddings_parser.add_argument(
        "--db-path",
        type=str,
        default=config.paper_db_path,
        help="Path to the SQLite database with papers",
    )
    embeddings_parser.add_argument(
        "--output",
        type=str,
        default=config.embedding_db_path,
        help=f"Output directory for ChromaDB vector database (default: {config.embedding_db_path})",
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
        "--embeddings-path",
        type=str,
        default=config.embedding_db_path,
        help=f"Path to ChromaDB vector database (default: {config.embedding_db_path})",
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
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file to resolve author names (optional)",
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
        help="Interactive RAG chat with NeurIPS papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Start an interactive chat session using RAG to answer questions about papers.",
    )
    chat_parser.add_argument(
        "--embeddings-path",
        type=str,
        default=config.embedding_db_path,
        help=f"Path to ChromaDB vector database (default: {config.embedding_db_path})",
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
        description="Start a Flask web server with a modern UI for exploring NeurIPS papers.",
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
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
