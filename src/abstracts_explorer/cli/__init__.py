"""
Command-line interface package.

The CLI is split into one module per command group; this package re-exports
the entry point and all command functions so ``abstracts_explorer.cli``
keeps a single import surface (the ``abstracts-explorer`` console script
points at ``abstracts_explorer.cli:main``).
"""

from abstracts_explorer.cli._common import (
    _build_embeddings_where_filters,
    _confirm,
    _print_header,
    _resolve_conference_arg,
    _resolve_registry_auth,
    _unexpected_error,
    _validate_embeddings_db,
    add_conference_year_args,
    setup_logging,
)
from abstracts_explorer.cli.clustering import (
    clear_clustering_cache_command,
    cluster_embeddings_command,
    pre_generate_clustering_command,
)
from abstracts_explorer.cli.data_mgmt import delete_data_command
from abstracts_explorer.cli.consistency import consistency_check_command
from abstracts_explorer.cli.download import (
    _download_single,
    _report_failed_downloads,
    download_command,
)
from abstracts_explorer.cli.embeddings import create_embeddings_command
from abstracts_explorer.cli.evaluation import (
    eval_clear_command,
    eval_generate_command,
    eval_results_command,
    eval_run_command,
    eval_verify_command,
)
from abstracts_explorer.cli.feedback import (
    feedback_browse_command,
    feedback_clear_command,
    feedback_list_command,
    feedback_stats_command,
)
from abstracts_explorer.cli.main import __version__, main
from abstracts_explorer.cli.pipeline import (
    _build_pre_process_clustering_args,
    _build_pre_process_embeddings_args,
    _clustering_worker,
    _download_into_queue,
    _embedding_worker,
    _post_process_worker,
    _pre_process_parallel,
    pre_process_command,
)
from abstracts_explorer.cli.registry import (
    _registry_download_has_existing_data,
    registry_delete_command,
    registry_download_command,
    registry_list_command,
    registry_upload_command,
)
from abstracts_explorer.cli.search_chat import chat_command, search_command
from abstracts_explorer.cli.servers import mcp_server_command, web_ui_command

__all__ = [
    "_build_embeddings_where_filters",
    "_build_pre_process_clustering_args",
    "_build_pre_process_embeddings_args",
    "_clustering_worker",
    "_confirm",
    "_download_into_queue",
    "_download_single",
    "_embedding_worker",
    "_post_process_worker",
    "_pre_process_parallel",
    "_print_header",
    "_registry_download_has_existing_data",
    "_report_failed_downloads",
    "_resolve_conference_arg",
    "_resolve_registry_auth",
    "_unexpected_error",
    "_validate_embeddings_db",
    "__version__",
    "add_conference_year_args",
    "chat_command",
    "clear_clustering_cache_command",
    "cluster_embeddings_command",
    "consistency_check_command",
    "create_embeddings_command",
    "delete_data_command",
    "download_command",
    "eval_clear_command",
    "eval_generate_command",
    "eval_results_command",
    "eval_run_command",
    "eval_verify_command",
    "feedback_browse_command",
    "feedback_clear_command",
    "feedback_list_command",
    "feedback_stats_command",
    "main",
    "mcp_server_command",
    "pre_generate_clustering_command",
    "pre_process_command",
    "registry_delete_command",
    "registry_download_command",
    "registry_list_command",
    "registry_upload_command",
    "search_command",
    "setup_logging",
    "web_ui_command",
]
