"""
Command-line interface package.

The CLI is split into one module per command group; this package re-exports
the entry point and all command functions so ``abstracts_explorer.cli``
keeps a single import surface (the ``abstracts-explorer`` console script
points at ``abstracts_explorer.cli:main``).

Re-exports are resolved lazily via :pep:`562` so importing this package does
not load every command module (and its heavy dependencies) on startup.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        eval_clear_results_command,
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

_LAZY_IMPORTS = {
    "_build_embeddings_where_filters": "abstracts_explorer.cli._common",
    "_confirm": "abstracts_explorer.cli._common",
    "_print_header": "abstracts_explorer.cli._common",
    "_resolve_conference_arg": "abstracts_explorer.cli._common",
    "_resolve_registry_auth": "abstracts_explorer.cli._common",
    "_unexpected_error": "abstracts_explorer.cli._common",
    "_validate_embeddings_db": "abstracts_explorer.cli._common",
    "add_conference_year_args": "abstracts_explorer.cli._common",
    "setup_logging": "abstracts_explorer.cli._common",
    "clear_clustering_cache_command": "abstracts_explorer.cli.clustering",
    "cluster_embeddings_command": "abstracts_explorer.cli.clustering",
    "pre_generate_clustering_command": "abstracts_explorer.cli.clustering",
    "delete_data_command": "abstracts_explorer.cli.data_mgmt",
    "consistency_check_command": "abstracts_explorer.cli.consistency",
    "_download_single": "abstracts_explorer.cli.download",
    "_report_failed_downloads": "abstracts_explorer.cli.download",
    "download_command": "abstracts_explorer.cli.download",
    "create_embeddings_command": "abstracts_explorer.cli.embeddings",
    "eval_clear_command": "abstracts_explorer.cli.evaluation",
    "eval_clear_results_command": "abstracts_explorer.cli.evaluation",
    "eval_generate_command": "abstracts_explorer.cli.evaluation",
    "eval_results_command": "abstracts_explorer.cli.evaluation",
    "eval_run_command": "abstracts_explorer.cli.evaluation",
    "eval_verify_command": "abstracts_explorer.cli.evaluation",
    "feedback_browse_command": "abstracts_explorer.cli.feedback",
    "feedback_clear_command": "abstracts_explorer.cli.feedback",
    "feedback_list_command": "abstracts_explorer.cli.feedback",
    "feedback_stats_command": "abstracts_explorer.cli.feedback",
    "__version__": "abstracts_explorer.cli.main",
    "main": "abstracts_explorer.cli.main",
    "_build_pre_process_clustering_args": "abstracts_explorer.cli.pipeline",
    "_build_pre_process_embeddings_args": "abstracts_explorer.cli.pipeline",
    "_clustering_worker": "abstracts_explorer.cli.pipeline",
    "_download_into_queue": "abstracts_explorer.cli.pipeline",
    "_embedding_worker": "abstracts_explorer.cli.pipeline",
    "_post_process_worker": "abstracts_explorer.cli.pipeline",
    "_pre_process_parallel": "abstracts_explorer.cli.pipeline",
    "pre_process_command": "abstracts_explorer.cli.pipeline",
    "_registry_download_has_existing_data": "abstracts_explorer.cli.registry",
    "registry_delete_command": "abstracts_explorer.cli.registry",
    "registry_download_command": "abstracts_explorer.cli.registry",
    "registry_list_command": "abstracts_explorer.cli.registry",
    "registry_upload_command": "abstracts_explorer.cli.registry",
    "chat_command": "abstracts_explorer.cli.search_chat",
    "search_command": "abstracts_explorer.cli.search_chat",
    "mcp_server_command": "abstracts_explorer.cli.servers",
    "web_ui_command": "abstracts_explorer.cli.servers",
}


def __getattr__(name: str) -> Any:
    """Lazily import a CLI command symbol on first access (PEP 562)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


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
    "eval_clear_results_command",
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
