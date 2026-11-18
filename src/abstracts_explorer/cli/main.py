"""
CLI entry point for abstracts-explorer.

Wires the argument parser for all subcommands (implemented in the sibling
modules of this package), parses arguments, and dispatches to the selected
command.
"""

import logging
import sys

import argcomplete


from abstracts_explorer.config import get_config


from abstracts_explorer.cli._common import setup_logging
from abstracts_explorer.cli.parsers import build_parser
from abstracts_explorer.cli.clustering import (
    clear_clustering_cache_command,
    cluster_embeddings_command,
    pre_generate_clustering_command,
)
from abstracts_explorer.cli.data_mgmt import delete_data_command
from abstracts_explorer.cli.download import download_command
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
from abstracts_explorer.cli.pipeline import pre_process_command
from abstracts_explorer.cli.registry import (
    registry_delete_command,
    registry_download_command,
    registry_list_command,
    registry_upload_command,
)
from abstracts_explorer.cli.search_chat import chat_command, search_command
from abstracts_explorer.cli.servers import mcp_server_command, web_ui_command

try:
    from abstracts_explorer._version import __version__  # noqa: F401  (re-exported)
except ImportError:
    from abstracts_explorer import __version__  # type: ignore[no-redef]  # noqa: F401

logger = logging.getLogger(__name__)


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

    parser, groups = build_parser(config)
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
            groups["clustering"].print_help()
            return 1
        if args.clustering_command == "run":
            return cluster_embeddings_command(args)
        elif args.clustering_command == "clear-cache":
            return clear_clustering_cache_command(args)
        elif args.clustering_command == "pre-generate":
            return pre_generate_clustering_command(args)
        else:
            groups["clustering"].print_help()
            return 1
    elif args.command == "mcp-server":
        return mcp_server_command(args)
    elif args.command == "eval":
        if not hasattr(args, "eval_command") or not args.eval_command:
            groups["eval"].print_help()
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
            groups["eval"].print_help()
            return 1
    elif args.command == "delete-data":
        return delete_data_command(args)
    elif args.command == "feedback":
        if not hasattr(args, "feedback_command") or not args.feedback_command:
            groups["feedback"].print_help()
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
            groups["feedback"].print_help()
            return 1
    elif args.command == "registry":
        if not hasattr(args, "registry_command") or not args.registry_command:
            groups["registry"].print_help()
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
            groups["registry"].print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
