"""
``papers search`` and ``papers chat`` commands

"""

import argparse
import logging
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager, EmbeddingsError
from abstracts_explorer.rag import RAGChat, RAGError


from abstracts_explorer.cli._common import (
    _print_header,
    _resolve_conference_arg,
    _unexpected_error,
    _validate_embeddings_db,
)

logger = logging.getLogger(__name__)


def search_command(args: argparse.Namespace) -> int:
    """
    Search the vector database for similar papers.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - query: Search query text
        - collection: Name of the ChromaDB collection
        - limit: Number of results to return
        - where: Metadata filter conditions
        - show_abstract: Whether to show paper abstracts
        - embedding_backend_url: URL for OpenAI-compatible embedding API
        - embedding_model: Name of the embedding model

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    config = get_config()

    # Validate embeddings database exists (for local paths only)
    if not _validate_embeddings_db(config):
        return 1

    # Resolve conference name once to canonical form
    conference = _resolve_conference_arg(getattr(args, "conference", None))
    year = getattr(args, "year", None)
    distance_threshold = getattr(args, "distance_threshold", 1.2)

    _print_header(
        "Semantic Search",
        f"Query: {args.query}",
        f"Embeddings: {config.embedding_db}",
        f"Collection: {args.collection}",
        f"Results: {args.limit}",
        f"Distance threshold: {distance_threshold}",
        f"Conference: {conference}" if conference else None,
        f"Year: {year}" if year is not None else None,
    )

    try:
        # Initialize embeddings manager
        with closing(
            EmbeddingsManager(
                backend_url=args.embedding_backend_url,
                model_name=args.embedding_model,
                collection_name=args.collection,
            )
        ) as em:
            # Test OpenAI API connection
            if not em.test_api_connection(verbose=True):
                return 1

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
                    print(f"❌ Could not parse filter '{args.where}': {e}", file=sys.stderr)
                    return 1

            # Perform search through the DatabaseManager search entry point
            print(f"\n🔍 Searching for: '{args.query}'...\n")
            database = DatabaseManager()
            try:
                papers, _ = database.search(
                    args.query,
                    limit=args.limit,
                    years=[year] if year is not None else None,
                    conferences=[conference] if conference else None,
                    distance_threshold=distance_threshold,
                    where_filters=where_filter,
                    embeddings_manager=em,
                    use_embeddings=True,
                )
            finally:
                database.close()

            if not papers:
                print("❌ No results found.")
                return 0

            num_results = len(papers)
            print(f"✅ Found {num_results} similar paper(s):\n")

            for i, paper in enumerate(papers):
                similarity = paper.get("similarity", 0.0)

                print(f"{i + 1}. Paper ID: {paper.get('uid', 'N/A')}")
                print(f"   Title: {paper.get('title', 'N/A')}")

                authors = paper.get("authors", [])
                authors_display = "; ".join(authors) if isinstance(authors, list) else str(authors)
                print(f"   Authors: {authors_display}")

                if paper.get("session"):
                    print(f"   Session: {paper.get('session')}")

                if paper.get("decision"):
                    print(f"   Decision: {paper.get('decision')}")

                if paper.get("topic"):
                    print(f"   Topic: {paper.get('topic')}")

                if paper.get("url") or paper.get("paper_url"):
                    print(f"   URL: {paper.get('url') or paper.get('paper_url')}")

                if paper.get("poster_position"):
                    print(f"   Poster Position: {paper.get('poster_position')}")

                print(f"   Similarity: {similarity:.4f}")

                if args.show_abstract and paper.get("abstract"):
                    abstract = paper["abstract"]
                    abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                    print(f"   Abstract: {abstract}")

                print()

            return 0

    except EmbeddingsError as e:
        print(f"\n❌ Search error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)


def chat_command(args: argparse.Namespace) -> int:
    """
    Interactive RAG chat with papers.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - collection: Name of the collection
        - llm_backend_url: OpenAI-compatible LLM API URL
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

        _print_header(
            "RAG Chat",
            f"Embeddings: {config.embedding_db}",
            f"Collection: {args.collection}",
            f"Chat Model: {args.model}",
            f"Embedding Model: {args.embedding_model}",
            f"API URL: {args.llm_backend_url}",
            f"Conference: {conference}" if conference else None,
            f"Year: {year}" if year is not None else None,
        )

        # Check embeddings exist (for local paths only)
        if not _validate_embeddings_db(config):
            return 1

        # Initialize embeddings manager
        with closing(
            EmbeddingsManager(
                collection_name=args.collection,
                backend_url=args.llm_backend_url,
                model_name=args.embedding_model,
            )
        ) as em:
            # Test OpenAI API connection
            if not em.test_api_connection(verbose=True, model_hint="A language model is loaded"):
                return 1

            # Get collection stats
            stats = em.get_collection_stats()
            print(f"\n📊 Loaded {stats['count']:,} papers from collection '{stats['name']}'")

            # Initialize database connection
            with closing(DatabaseManager()) as db:
                # Build conference/year context for RAG queries (case-insensitive via DB lookup)
                chat_conferences: Optional[List[str]] = None
                chat_years: Optional[List[int]] = None
                if conference:
                    chat_conferences = [conference]
                if year is not None:
                    chat_years = [year]

                # Initialize RAG chat
                chat = RAGChat(
                    embeddings_manager=em,
                    database=db,
                    backend_url=args.llm_backend_url,
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
                        if result.get("metadata", {}).get("guard", {}).get("blocked"):
                            print("Assistant:")
                        else:
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

                return 0

    except RAGError as e:
        print(f"\n❌ RAG error: {e}", file=sys.stderr)
        return 1
    except EmbeddingsError as e:
        print(f"\n❌ Embeddings error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e)
