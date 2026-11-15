"""
Argument parser construction for the CLI.

:func:`build_parser` wires every subcommand and returns the parser plus the
nested subparser groups used for help output.
"""

import argparse

from abstracts_explorer.cli._common import add_conference_year_args
from abstracts_explorer.config import Config
from abstracts_explorer.plugins import list_plugins

try:
    from abstracts_explorer._version import __version__
except ImportError:
    from abstracts_explorer import __version__  # type: ignore[no-redef]


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
        ``(parser, groups)`` where *groups* maps nested subcommand names
        (clustering/eval/feedback/registry) to their subparsers.
    """
    parser = argparse.ArgumentParser(
        prog="abstracts-explorer",
        description="Abstracts Explorer - Tools for working with conference abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Download NeurIPS 2025 data and create database
      abstracts-explorer download neurips --year 2025

      # Generate embeddings for all papers
      abstracts-explorer create-embeddings

      # Search for similar papers
      abstracts-explorer search "graph neural networks for molecular generation"

      # Search with custom settings
      abstracts-explorer search "deep learning transformers" \\
    --n-results 10 \\
    --show-abstract

      # Search with metadata filter
      abstracts-explorer search "reinforcement learning" \\
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
    download_description = """
    Download papers from various sources using plugins.

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
      abstracts-explorer download

      # Download all years for a specific conference
      abstracts-explorer download --conference neurips

      # Download a specific conference and year
      abstracts-explorer download --conference neurips --year 2025

      # Load CHI 2024 papers from a pre-downloaded JSON
      abstracts-explorer download --conference chi --year 2024 --input-file chi_2024_program.json

      # List available plugins
      abstracts-explorer download --list-plugins
        """
    download_parser = subparsers.add_parser(
        "download",
        help="Download conference data and create database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=download_description,
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
        "--request-delay",
        type=float,
        default=None,
        help="Delay before each HTTP request for plugins that support it (AGU).",
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
        "--backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible API (default: {config.llm_backend_url})",
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
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Recreate embeddings after an embedding model mismatch without prompting",
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
        "--backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible API (default: {config.llm_backend_url})",
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
        "--backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible API (default: {config.llm_backend_url})",
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
        help="Pre-generate clustering results for the web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
    Pre-generate clustering results using agglomerative clustering
    (ward linkage, distance threshold 150) with t-SNE dimensionality
    reduction.  Results are persisted to the database cache so that
    the web UI serves them instantly.

    Without any arguments, generates clustering for every conference in
    the database combined with each individual year.

    With --conference only, generates for that conference for each
    individual year.

    With --conference and --year, generates for that specific conference
    and year only.

    Examples:
      # Pre-generate for all conference/year combinations (default)
      abstracts-explorer clustering pre-generate

      # Pre-generate for a specific conference (each year)
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
      # Start MCP server on default port (8000) using streamable HTTP
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
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="Transport method: streamable-http (HTTP) or stdio (default: streamable-http)",
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
        "--backend-url",
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
        "--backend-url",
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
    Run the full pre-processing pipeline:
      1. download          - Download conference data
      2. create-embeddings - Generate embeddings for abstracts
      3. clustering pre-generate - Pre-generate clustering results for the web UI

    By default the pipeline is parallel: each conference/year unit is embedded and
    clustered as soon as its download finishes, while the remaining downloads keep
    running.  Embeddings and clustering also overlap, so clustering of one unit runs
    while embeddings for the next unit are being created.  Use
    --no-overlap-post-process to run embeddings and clustering in a single worker.

    All steps receive the same --conference and --year filters.  The embedding and
    clustering options below are forwarded to the respective steps.

    Examples:
      # Process all conferences and years (parallel pipeline)
      abstracts-explorer pre-process

      # Process only NeurIPS (all years)
      abstracts-explorer pre-process --conference neurips

      # Process NeurIPS 2025 only
      abstracts-explorer pre-process --conference neurips --year 2025

      # Re-process, overwriting existing data
      abstracts-explorer pre-process --conference neurips --force

      # Recreate embeddings after switching embedding models, without prompting
      abstracts-explorer pre-process --ignore-embedding-model-mismatch --model my-new-model

      # Do not overlap embeddings and clustering
      abstracts-explorer pre-process --no-overlap-post-process
        """,
    )
    add_conference_year_args(pre_process_parser)
    pre_process_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reset embeddings/clustering even if they already exist",
    )
    pre_process_parser.add_argument(
        "--no-overlap-post-process",
        action="store_true",
        help="Run embeddings and clustering in a single worker instead of overlapping them",
    )
    pre_process_parser.add_argument(
        "--collection",
        type=str,
        default=config.collection_name,
        help=f"Name for the ChromaDB collection (default: {config.collection_name})",
    )
    pre_process_parser.add_argument(
        "--backend-url",
        type=str,
        default=config.llm_backend_url,
        help=f"URL for OpenAI-compatible API (default: {config.llm_backend_url})",
    )
    pre_process_parser.add_argument(
        "--model",
        type=str,
        default=config.embedding_model,
        help=f"Name of the embedding model (default: {config.embedding_model})",
    )
    pre_process_parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=config.requests_per_minute,
        help=f"Maximum number of API requests per minute, 0 to disable rate limiting (default: {config.requests_per_minute})",
    )
    pre_process_parser.add_argument(
        "--ignore-embedding-model-mismatch",
        action="store_true",
        help="Recreate embeddings after an embedding model mismatch without prompting",
    )

    groups = {
        "clustering": clustering_parser,
        "eval": eval_parser,
        "feedback": feedback_parser,
        "registry": registry_parser,
    }
    return parser, groups
