"""
Argument parser construction for the CLI.

:func:`build_parser` wires every subcommand and returns the parser plus the
nested subparser groups used for help output.
"""

import argparse
from pathlib import Path

from abstracts_explorer.cli._common import add_conference_year_args
from abstracts_explorer.config import Config
from abstracts_explorer.plugins import list_plugins

try:
    from abstracts_explorer._version import __version__
except ImportError:
    from abstracts_explorer import __version__  # type: ignore[no-redef]

_FORMATTER = argparse.RawDescriptionHelpFormatter


def _add_group(
    parent: argparse._SubParsersAction,
    name: str,
    help: str,
    *,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Add a noun-group subparser and register it for help dispatch."""
    parser = parent.add_parser(
        name,
        help=help,
        formatter_class=_FORMATTER,
        description=description,
        epilog=epilog,
    )
    parser.set_defaults(_selected_group=parser)
    return parser


def _add_leaf(
    parent: argparse._SubParsersAction,
    name: str,
    handler: str,
    path: tuple[str, ...],
    *,
    help: str,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Add a leaf subcommand and bind its handler path."""
    parser = parent.add_parser(
        name,
        help=help,
        formatter_class=_FORMATTER,
        description=description,
        epilog=epilog,
    )
    parser.set_defaults(_handler=handler, _command_path=path)
    return parser


def build_parser(config: Config) -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """
    Build the CLI argument parser.

    Parameters
    ----------
    config : Config
        Loaded configuration providing command defaults.

    Returns
    -------
    tuple
        ``(parser, groups)`` where *groups* maps top-level command names to
        their subparsers for help output.
    """
    parser = argparse.ArgumentParser(
        prog="abstracts-explorer",
        description="Abstracts Explorer - Tools for working with conference abstracts",
        formatter_class=_FORMATTER,
        epilog="""
    Examples:
      # Download NeurIPS 2025 data and create database
      abstracts-explorer data download --conference neurips --year 2025

      # Generate embeddings for all papers
      abstracts-explorer embeddings create

      # Search for similar papers
      abstracts-explorer papers search "graph neural networks for molecular generation"

      # Search with custom settings
      abstracts-explorer papers search "deep learning transformers" \\
    --limit 10 --show-abstract

      # Start the web UI
      abstracts-explorer serve web
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        default=None,
        help=("Refuse database and embedding writes and allow multiple read-only " "instances (env: READ_ONLY)."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    groups: dict[str, argparse.ArgumentParser] = {}

    # -------------------------------------------------------------------------
    # data
    # -------------------------------------------------------------------------
    download_description = """
    Download papers from various sources using plugins and store them in the database.

    Without arguments, downloads ALL conferences for ALL available years
    (plugins that require manual input are skipped).

    Available plugins:
    """
    for plugin_meta in list_plugins():
        years = plugin_meta.get("supported_years") or []
        year_note = f" ({min(years)}-{max(years)})" if years else ""
        download_description += f"  {plugin_meta['name']:<8} - {plugin_meta['description']}{year_note}\n"
    download_description += """
    Examples:
      # Download ALL conference data for ALL available years
      abstracts-explorer data download

      # Download all years for a specific conference
      abstracts-explorer data download --conference neurips

      # Download a specific conference and year
      abstracts-explorer data download --conference neurips --year 2025

      # Load CHI 2024 papers from a pre-downloaded JSON
      abstracts-explorer data download --conference chi --year 2024 --input-file chi_2024_program.json

      # List available plugins
      abstracts-explorer data download --list-plugins
        """
    data_parser = _add_group(
        subparsers,
        "data",
        "Download, delete, and check conference data",
        description="Acquire and manage conference data in the paper database and embedding store.",
    )
    groups["data"] = data_parser
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data sub-commands")

    download_parser = _add_leaf(
        data_subparsers,
        "download",
        "download_command",
        ("data", "download"),
        help="Download conference data and create database",
        description=download_description,
    )
    download_parser.add_argument(
        "--output",
        type=str,
        default=str(Path(config.data_dir) / "abstracts.json"),
        help=f"Output path for intermediate JSON file (default: {Path(config.data_dir) / 'abstracts.json'})",
    )
    add_conference_year_args(download_parser)
    download_parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    download_parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available downloader plugins and exit",
    )
    download_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers for fetching data (plugin-specific default when omitted)",
    )
    download_parser.add_argument(
        "--request-delay",
        type=float,
        default=None,
        help="Delay before each HTTP request for plugins that support it (AGU, NeurIPS, ICML, ICLR).",
    )
    download_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard an interrupted download checkpoint and start fresh (AGU).",
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
    download_parser.add_argument(
        "--openreview-email",
        type=str,
        default=None,
        dest="openreview_email",
        help="OpenReview account email for the ICLR downloader (env: OPENREVIEW_EMAIL).",
    )
    download_parser.add_argument(
        "--openreview-password",
        type=str,
        default=None,
        dest="openreview_password",
        help="OpenReview account password for the ICLR downloader (env: OPENREVIEW_PASSWORD).",
    )
    download_parser.add_argument(
        "--openreview-token",
        type=str,
        default=None,
        dest="openreview_token",
        help=(
            "Existing OpenReview bearer token for the ICLR downloader "
            "(env: OPENREVIEW_TOKEN). Takes precedence over email/password."
        ),
    )

    delete_parser = _add_leaf(
        data_subparsers,
        "delete",
        "delete_data_command",
        ("data", "delete"),
        help="Delete all data for a specific conference/year",
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
      abstracts-explorer data delete --conference neurips --year 2024

      # Delete without interactive confirmation
      abstracts-explorer data delete --conference iclr --year 2023 --yes
        """,
    )
    delete_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        required=True,
        help="Conference to delete (required). Case-insensitive.",
    )
    delete_parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of conference/workshop to delete (required).",
    )
    delete_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )

    check_parser = _add_leaf(
        data_subparsers,
        "check",
        "consistency_check_command",
        ("data", "check"),
        help="Check that papers, embeddings, and clustering data are consistent",
        description="""
    Compare the paper database, ChromaDB embeddings, and clustering cache.

    The check reports missing, orphaned, or stale embeddings, embedding-model
    mismatches, missing or duplicated clustering cache entries, cache payloads
    that do not match the paper database, and unused ChromaDB collections.

    Use --fix to delete duplicate clustering cache entries and cache entries
    for scopes that no longer have papers.  Other findings are reported with a
    suggested command; embeddings and missing caches are never modified
    automatically.

    Examples:
      # Check all data
      abstracts-explorer data check

      # Check one conference/year and print JSON
      abstracts-explorer data check --conference NeurIPS --year 2025 --json

      # Remove duplicate/stale clustering cache rows
      abstracts-explorer data check --fix
        """,
    )
    add_conference_year_args(check_parser)
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of human-readable text.",
    )
    check_parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete duplicate clustering cache rows and cache rows for scopes without papers.",
    )

    # -------------------------------------------------------------------------
    # embeddings
    # -------------------------------------------------------------------------
    embeddings_parser = _add_group(
        subparsers,
        "embeddings",
        "Create and manage embeddings",
        description="Create embeddings for paper abstracts using an OpenAI-compatible API.",
    )
    groups["embeddings"] = embeddings_parser
    embeddings_subparsers = embeddings_parser.add_subparsers(
        dest="embeddings_command", help="Embeddings sub-commands"
    )

    embeddings_create_parser = _add_leaf(
        embeddings_subparsers,
        "create",
        "create_embeddings_command",
        ("embeddings", "create"),
        help="Generate embeddings for abstracts",
        description="""
    Generate embeddings for paper abstracts using an OpenAI-compatible API.

    Without additional arguments, embeds all conference data for all
    available years.

    With --conference and/or --year, only the matching papers are embedded.

    Examples:
      # Embed all papers (default)
      abstracts-explorer embeddings create

      # Embed only NeurIPS papers
      abstracts-explorer embeddings create --conference NeurIPS

      # Embed only NeurIPS 2024 papers
      abstracts-explorer embeddings create --conference NeurIPS --year 2024

      # Embed only papers from 2025 (all conferences)
      abstracts-explorer embeddings create --year 2025

        """,
    )
    add_conference_year_args(embeddings_create_parser)
    embeddings_create_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name for the ChromaDB collection (default: {config.collection_name})",
    )
    embeddings_create_parser.add_argument(
        "--embedding-backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible embedding API (default: {config.llm_backend_url})",
    )
    embeddings_create_parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    embeddings_create_parser.add_argument(
        "--force",
        action="store_true",
        help="Reset collection if it already exists",
    )
    embeddings_create_parser.add_argument(
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Recreate embeddings after an embedding model mismatch without prompting",
    )
    embeddings_create_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )

    # -------------------------------------------------------------------------
    # clustering
    # -------------------------------------------------------------------------
    clustering_parser = _add_group(
        subparsers,
        "clustering",
        "Cluster embeddings and manage the clustering cache",
        description="""
    Clustering-related commands.

    Uses BERTopic with Agglomerative clustering (ward linkage, distance
    threshold 0.5) on 5-D UMAP embeddings and 2-D UMAP visualization.

    Examples:
      # Cluster with default settings
      abstracts-explorer clustering run

      # Populate the cache for the web UI
      abstracts-explorer clustering cache populate

      # Clear the cache
      abstracts-explorer clustering cache clear
        """,
    )
    groups["clustering"] = clustering_parser
    clustering_subparsers = clustering_parser.add_subparsers(
        dest="clustering_command", help="Clustering sub-commands"
    )

    cluster_parser = _add_leaf(
        clustering_subparsers,
        "run",
        "cluster_embeddings_command",
        ("clustering", "run"),
        help="Cluster embeddings for visualization",
        description="Perform BERTopic clustering with UMAP dimensionality reduction on paper embeddings.",
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
    cluster_parser.add_argument(
        "--no-llm-labels",
        action="store_true",
        default=False,
        help="Use c-TF-IDF keywords instead of LLM-generated labels",
    )
    cluster_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )
    add_conference_year_args(cluster_parser)

    clustering_cache_parser = _add_group(
        clustering_subparsers,
        "cache",
        "Manage the clustering cache",
        description="Populate or clear clustering results stored in the database cache.",
    )
    cache_subparsers = clustering_cache_parser.add_subparsers(dest="cache_command", help="Cache sub-commands")

    cache_populate_parser = _add_leaf(
        cache_subparsers,
        "populate",
        "pre_generate_clustering_command",
        ("clustering", "cache", "populate"),
        help="Compute clustering results and store them in the cache",
        description="""
    Compute clustering results and store them in the database cache so the
    web UI can serve them instantly.

    Uses BERTopic with Agglomerative clustering (ward linkage, distance
    threshold 0.5) on 5-D UMAP embeddings.

    Without any arguments, populates the cache for every conference in the
    database combined with each individual year.

    With --conference only, populates the cache for that conference for each
    individual year.

    With --conference and --year, populates the cache for that specific
    conference and year only.

    Examples:
      # Populate the cache for all conference/year combinations (default)
      abstracts-explorer clustering cache populate

      # Populate for a specific conference (each year)
      abstracts-explorer clustering cache populate --conference NeurIPS

      # Populate for a specific conference and year
      abstracts-explorer clustering cache populate --conference NeurIPS --year 2024

      # Force recompute
      abstracts-explorer clustering cache populate --force

      # Use c-TF-IDF keywords instead of LLM labels
      abstracts-explorer clustering cache populate --no-llm-labels
        """,
    )
    cache_populate_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    add_conference_year_args(cache_populate_parser)
    cache_populate_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recompute even if cache already exists",
    )
    cache_populate_parser.add_argument(
        "--no-llm-labels",
        action="store_true",
        default=False,
        help="Use c-TF-IDF keywords instead of LLM-generated labels",
    )
    cache_populate_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )

    cache_clear_parser = _add_leaf(
        cache_subparsers,
        "clear",
        "clear_clustering_cache_command",
        ("clustering", "cache", "clear"),
        help="Clear clustering cache from the database",
        description="""
    Clear clustering cache from the database.

    This is useful when embeddings change or cache becomes stale.
    You can optionally filter by embedding model to clear only specific entries.

    Examples:
      # Clear all clustering cache entries
      abstracts-explorer clustering cache clear

      # Clear cache for a specific embedding model
      abstracts-explorer clustering cache clear --embedding-model text-embedding-3-large
        """,
    )
    cache_clear_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Only clear cache for this embedding model (optional)",
    )

    # -------------------------------------------------------------------------
    # papers
    # -------------------------------------------------------------------------
    papers_parser = _add_group(
        subparsers,
        "papers",
        "Search and chat with papers",
        description="Query the paper database using semantic search or RAG chat.",
    )
    groups["papers"] = papers_parser
    papers_subparsers = papers_parser.add_subparsers(dest="papers_command", help="Papers sub-commands")

    search_parser = _add_leaf(
        papers_subparsers,
        "search",
        "search_command",
        ("papers", "search"),
        help="Search the vector database for similar papers",
    )
    search_parser.add_argument("query", type=str, help="Search query text")
    search_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    search_parser.add_argument(
        "--limit",
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
        "--embedding-backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible embedding API (default: {config.llm_backend_url})",
    )
    search_parser.add_argument(
        "--embedding-model",
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

    chat_parser = _add_leaf(
        papers_subparsers,
        "chat",
        "chat_command",
        ("papers", "chat"),
        help="Interactive RAG chat with papers",
        description="Start an interactive chat session using RAG to answer questions about papers.",
    )
    chat_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name of the ChromaDB collection (default: {config.collection_name})",
    )
    chat_parser.add_argument(
        "--llm-backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible LLM API (default: {config.llm_backend_url})",
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

    # -------------------------------------------------------------------------
    # pipeline
    # -------------------------------------------------------------------------
    pipeline_parser = _add_group(
        subparsers,
        "pipeline",
        "Run the full data processing pipeline",
        description="Orchestrate download, embeddings, and clustering cache population.",
    )
    groups["pipeline"] = pipeline_parser
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command", help="Pipeline sub-commands")

    pipeline_run_parser = _add_leaf(
        pipeline_subparsers,
        "run",
        "pre_process_command",
        ("pipeline", "run"),
        help="Download, embed, and populate the clustering cache",
        description="""
    Run the full data processing pipeline:
      1. data download            - Download conference data
      2. embeddings create        - Generate embeddings for abstracts
      3. clustering cache populate - Compute clustering results for the web UI

    By default the pipeline is parallel: each conference/year unit is embedded and
    clustered as soon as its download finishes, while the remaining downloads keep
    running.  Embeddings and clustering also overlap, so clustering of one unit runs
    while embeddings for the next unit are being created.  Use
    --serial-post-process to run embeddings and clustering in a single worker.

    All steps receive the same --conference and --year filters.  The embedding and
    clustering options below are forwarded to the respective steps.

    Examples:
      # Process all conferences and years (parallel pipeline)
      abstracts-explorer pipeline run

      # Process only NeurIPS (all years)
      abstracts-explorer pipeline run --conference neurips

      # Process NeurIPS 2025 only
      abstracts-explorer pipeline run --conference neurips --year 2025

      # Re-process, overwriting existing data
      abstracts-explorer pipeline run --conference neurips --force

      # Recreate embeddings after switching embedding models, without prompting
      abstracts-explorer pipeline run --ignore-embedding-model-mismatch --embedding-model my-new-model

      # Use c-TF-IDF keywords instead of LLM labels
      abstracts-explorer pipeline run --no-llm-labels

      # Do not overlap embeddings and clustering
      abstracts-explorer pipeline run --serial-post-process
        """,
    )
    add_conference_year_args(pipeline_run_parser)
    pipeline_run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reset embeddings/clustering even if they already exist",
    )
    pipeline_run_parser.add_argument(
        "--no-llm-labels",
        action="store_true",
        help="Use c-TF-IDF keywords instead of LLM-generated labels",
    )
    pipeline_run_parser.add_argument(
        "--serial-post-process",
        action="store_true",
        help="Run embeddings and clustering in a single worker instead of overlapping them",
    )
    pipeline_run_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name for the ChromaDB collection (default: {config.collection_name})",
    )
    pipeline_run_parser.add_argument(
        "--embedding-backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible embedding API (default: {config.llm_backend_url})",
    )
    pipeline_run_parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    pipeline_run_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )
    pipeline_run_parser.add_argument(
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Recreate embeddings after an embedding model mismatch without prompting",
    )

    # -------------------------------------------------------------------------
    # registry
    # -------------------------------------------------------------------------
    registry_parser = _add_group(
        subparsers,
        "registry",
        "Push/pull data to/from OCI-compatible container registries",
        description="""
    Upload and download paper databases, embeddings, and clustering cache
    to/from OCI-compatible container registries like GitHub Container Registry.

    Data is always uploaded/downloaded as a complete unit per conference and year
    to prevent inconsistent data between instances.  When --year is omitted,
    all available years for the conference are uploaded/downloaded together
    with each year stored as its own pair of OCI layers.

    Use --conference all to push/pull all available conferences at once.

    Examples:
      # Push NeurIPS 2024 data to GitHub Container Registry
      abstracts-explorer registry push --conference neurips --year 2024 \\
    --repository ghcr.io/thawn/abstracts-data

      # Push all NeurIPS years
      abstracts-explorer registry push --conference neurips \\
    --repository ghcr.io/thawn/abstracts-data

      # Pull NeurIPS 2024 data from registry
      abstracts-explorer registry pull --conference neurips --year 2024 \\
    --repository ghcr.io/thawn/abstracts-data

      # List available tags
      abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data

      # Inspect a specific tag
      abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data \\
    --tag neurips-2024
        """,
    )
    groups["registry"] = registry_parser
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", help="Registry sub-commands")

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

    registry_push_parser = _add_leaf(
        registry_subparsers,
        "push",
        "registry_upload_command",
        ("registry", "push"),
        help="Upload data for a conference (and optionally a specific year) to registry",
    )
    _add_registry_args(registry_push_parser)
    registry_push_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Conference name (e.g., neurips, iclr) or 'all' for all conferences (default: all).",
    )
    registry_push_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Conference year (e.g., 2024). If omitted, all available years are uploaded.",
    )
    registry_push_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag (default: derived from conference, year and embedding model)",
    )
    registry_push_parser.add_argument(
        "--only-tags",
        action="store_true",
        help="Do not upload data; re-tag the latest existing artifact for every tag base under the "
        "current version. Use when only the abstracts-explorer version changed but the data did not.",
    )
    registry_push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --only-tags, preview which tags would be created without modifying the registry.",
    )
    registry_push_parser.add_argument(
        "--force",
        action="store_true",
        help="With --only-tags, overwrite existing tags for the installed version.",
    )

    registry_update_conference_image_parser = _add_leaf(
        registry_subparsers,
        "update-conference-image",
        "registry_update_conference_image_command",
        ("registry", "update-conference-image"),
        help="Rebuild the whole-conference image from existing per-year artifacts without re-uploading data",
        description="""
    Rebuild the whole-conference registry image (the conference-level tag that
    contains every year) from the existing per-year artifacts.

    Only a small config blob and the image manifest are pushed; data layers for
    existing years are reused by digest, so no data is re-uploaded. Use this
    after pushing a new year, or to refresh the whole-conference image after
    other per-year changes.
        """,
    )
    _add_registry_args(registry_update_conference_image_parser)
    registry_update_conference_image_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        required=True,
        help="Conference name (e.g., neurips). A specific conference is required.",
    )
    registry_update_conference_image_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (defaults to the model stored in the local database).",
    )

    registry_pull_parser = _add_leaf(
        registry_subparsers,
        "pull",
        "registry_download_command",
        ("registry", "pull"),
        help="Download data for a conference (and optionally a specific year) from registry",
    )
    _add_registry_args(registry_pull_parser)
    registry_pull_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Conference name (e.g., neurips, iclr) or 'all' for all available tags (default: all).",
    )
    registry_pull_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Conference year (e.g., 2024). If omitted, all years in the artifact are downloaded.",
    )
    registry_pull_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag (default: derived from conference, year and embedding model)",
    )
    registry_pull_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name for tag derivation. "
        "If omitted, read from local database metadata or EMBEDDING_MODEL env var.",
    )
    registry_pull_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    registry_pull_parser.add_argument(
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Proceed even if the downloaded artifact uses a different embedding model than configured. "
        "Only use this option when the mismatch is caused by the same model having different names on "
        "different backends (e.g. LM Studio vs. Ollama). "
        "The local embedding model metadata will be updated to match the configured model after download.",
    )

    registry_list_parser = _add_leaf(
        registry_subparsers,
        "list",
        "registry_list_command",
        ("registry", "list"),
        help="List available tags in registry",
    )
    _add_registry_args(registry_list_parser)
    registry_list_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Inspect a specific tag (shows details instead of listing all tags)",
    )

    registry_delete_parser = _add_leaf(
        registry_subparsers,
        "delete",
        "registry_delete_command",
        ("registry", "delete"),
        help="Delete registry package versions below a given version",
        description="""
    Delete registry package versions whose OCI tag version is strictly below a given threshold,
    and/or untagged (dangling) versions.

    Uses the GitHub Packages API to list all versions of the container package and permanently
    deletes those that are outdated.  Untagged versions are only deleted with --untagged and are
    verified immediately before deletion (re-checked as still untagged and confirmed to be
    orphaned data artifacts, not indexes or attestations).

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

      # Delete untagged (dangling) versions only
      abstracts-explorer registry delete --untagged \\
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
        default=None,
        metavar="VERSION",
        help="Delete all versions strictly below this version (e.g., '0.4.0'). Must be a valid "
        "PEP 440 version. At least one of --below-version or --untagged is required.",
    )
    registry_delete_parser.add_argument(
        "--conference",
        "-c",
        type=str,
        default=None,
        help="Only delete versions whose tags belong to this conference (optional; all conferences if omitted).",
    )
    registry_delete_parser.add_argument(
        "--untagged",
        action="store_true",
        help="Also delete untagged (dangling) versions. Each is verified (still untagged, "
        "an orphaned data artifact) before deletion. Cannot be combined with --conference.",
    )
    registry_delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which versions would be deleted without actually deleting them.",
    )
    registry_delete_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    # -------------------------------------------------------------------------
    # eval
    # -------------------------------------------------------------------------
    eval_parser = _add_group(
        subparsers,
        "eval",
        "Automatic evaluation of the RAG system",
        description="""
    Automatic evaluation of the RAG system.

    Examples:
      # Generate Q/A pairs for all MCP tools
      abstracts-explorer eval generate

      # Generate 5 pairs per tool with 2 follow-ups each
      abstracts-explorer eval generate --n-pairs 5 --n-followups 2

      # Verify a random sample of 10 pairs
      abstracts-explorer eval verify --sample 10

      # Run evaluation
      abstracts-explorer eval run

      # Show latest results with details
      abstracts-explorer eval list --detail

      # Clear all accepted pairs without confirmation
      abstracts-explorer eval clear-pairs --yes

      # Delete results for a specific run without confirmation
      abstracts-explorer eval clear-results --run-id RUN_ID --yes
        """,
    )
    groups["eval"] = eval_parser
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", help="Evaluation sub-commands")

    eval_gen_parser = _add_leaf(
        eval_subparsers,
        "generate",
        "eval_generate_command",
        ("eval", "generate"),
        help="Generate evaluation Q/A pairs using an LLM",
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
    followup_group = eval_gen_parser.add_mutually_exclusive_group()
    followup_group.add_argument(
        "--no-followups",
        action="store_true",
        help="Disable follow-up question generation",
    )
    followup_group.add_argument(
        "--n-followups",
        type=int,
        default=None,
        help="Number of follow-up turns per initial pair (default: 1)",
    )
    eval_gen_parser.add_argument(
        "--model",
        type=str,
        default=config.chat_model,
        help=f"Chat model for generation (default: {config.chat_model})",
    )
    eval_gen_parser.add_argument(
        "--llm-backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"LLM backend URL (default: {config.llm_backend_url})",
    )

    eval_verify_parser = _add_leaf(
        eval_subparsers,
        "verify",
        "eval_verify_command",
        ("eval", "verify"),
        help="Interactively verify, edit, or delete Q/A pairs",
    )
    verify_group = eval_verify_parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of pairs to randomly sample for review",
    )
    verify_group.add_argument(
        "--all",
        action="store_true",
        help="Review all pairs (including already verified)",
    )

    eval_run_parser = _add_leaf(
        eval_subparsers,
        "run",
        "eval_run_command",
        ("eval", "run"),
        help="Run evaluation on stored Q/A pairs",
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
        "--llm-backend-url",
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

    eval_list_parser = _add_leaf(
        eval_subparsers,
        "list",
        "eval_results_command",
        ("eval", "list"),
        help="Browse evaluation results",
    )
    eval_list_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to display (default: latest)",
    )
    eval_list_parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed per-pair results",
    )
    eval_list_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N results for browsing",
    )

    eval_clear_pairs_parser = _add_leaf(
        eval_subparsers,
        "clear-pairs",
        "eval_clear_command",
        ("eval", "clear-pairs"),
        help="Delete all accepted (verified) Q/A pairs",
    )
    eval_clear_pairs_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    eval_clear_results_parser = _add_leaf(
        eval_subparsers,
        "clear-results",
        "eval_clear_results_command",
        ("eval", "clear-results"),
        help="Delete stored evaluation results for a run",
    )
    eval_clear_results_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to delete (default: latest)",
    )
    eval_clear_results_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # -------------------------------------------------------------------------
    # feedback
    # -------------------------------------------------------------------------
    feedback_parser = _add_group(
        subparsers,
        "feedback",
        "Explore user feedback (chat donations and data donations)",
        description="""
    Explore user feedback collected by the web UI.

    Two types of feedback are tracked:
      - Chat donations: anonymized chat transcripts with thumbs up/down ratings.
      - Data donations: anonymized interesting-paper ratings (1-5 priority).

    Examples:
      # Show summary statistics
      abstracts-explorer feedback stats

      # List all chat donations
      abstracts-explorer feedback list --type chat

      # List only thumbs-down chat donations
      abstracts-explorer feedback list --type chat --rating down

      # Interactively browse chat donations
      abstracts-explorer feedback browse --type chat

      # Delete only chat donations, no prompt
      abstracts-explorer feedback clear --type chat --yes
        """,
    )
    groups["feedback"] = feedback_parser
    feedback_subparsers = feedback_parser.add_subparsers(dest="feedback_command", help="Feedback sub-commands")

    _add_leaf(
        feedback_subparsers,
        "stats",
        "feedback_stats_command",
        ("feedback", "stats"),
        help="Show summary statistics for all user feedback",
    )

    feedback_list_parser = _add_leaf(
        feedback_subparsers,
        "list",
        "feedback_list_command",
        ("feedback", "list"),
        help="List individual feedback entries",
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

    feedback_browse_parser = _add_leaf(
        feedback_subparsers,
        "browse",
        "feedback_browse_command",
        ("feedback", "browse"),
        help="Interactively browse individual feedback entries",
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

    feedback_clear_parser = _add_leaf(
        feedback_subparsers,
        "clear",
        "feedback_clear_command",
        ("feedback", "clear"),
        help="Delete feedback data",
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

    # -------------------------------------------------------------------------
    # serve
    # -------------------------------------------------------------------------
    serve_parser = _add_group(
        subparsers,
        "serve",
        "Start the web UI or MCP server",
        description="Run a long-lived server exposing the papers over HTTP or MCP.",
    )
    groups["serve"] = serve_parser
    serve_subparsers = serve_parser.add_subparsers(dest="serve_command", help="Server sub-commands")

    web_parser = _add_leaf(
        serve_subparsers,
        "web",
        "web_ui_command",
        ("serve", "web"),
        help="Start the web interface",
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
        default=5001,
        help="Port to bind to (default: 5001)",
    )
    web_parser.add_argument(
        "--dev",
        action="store_true",
        help="Use Flask development server instead of production server (Waitress). Note: Use -vv to enable debug mode with any server.",
    )
    web_parser.add_argument(
        "--threads",
        type=int,
        default=config.web_ui_threads,
        help=(
            "Number of Waitress worker threads "
            f"(default: WEB_UI_THREADS or {config.web_ui_threads}). "
            "Must be >= 1. Ignored when --dev is set."
        ),
    )

    mcp_parser = _add_leaf(
        serve_subparsers,
        "mcp",
        "mcp_server_command",
        ("serve", "mcp"),
        help="Start MCP server for cluster analysis",
        description="""
    Start a Model Context Protocol (MCP) server for cluster analysis.

    The MCP server exposes tools that allow LLM-based assistants to:
    - Get most frequently mentioned topics from clusters
    - Analyze how topics evolved over years for conferences
    - Find recent developments in specific topics
    - Generate cluster visualizations

    Examples:
      # Start MCP server on default port (8000) using streamable HTTP
      abstracts-explorer serve mcp

      # Start on custom host and port
      abstracts-explorer serve mcp --host 0.0.0.0 --port 8080

      # Use stdio transport (for local CLI integration)
      abstracts-explorer serve mcp --transport stdio
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
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport method: streamable-http (HTTP) or stdio (default: streamable-http)",
    )

    return parser, groups
