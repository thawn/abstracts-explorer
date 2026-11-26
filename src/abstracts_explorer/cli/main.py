"""
CLI entry point for abstracts-explorer.

Wires the argument parser for all subcommands (implemented in the sibling
modules of this package), parses arguments, and dispatches to the selected
command.

Command functions are resolved lazily from :mod:`abstracts_explorer.cli` so
that ``--help`` and individual commands only import what they need.
"""

import logging
import sys
from typing import Any

import argcomplete


from abstracts_explorer.config import get_config


from abstracts_explorer.cli._common import _command_writes_storage, setup_logging
from abstracts_explorer.cli.parsers import build_parser

try:
    from abstracts_explorer._version import __version__  # noqa: F401  (re-exported)
except ImportError:
    from abstracts_explorer import __version__  # type: ignore[no-redef]  # noqa: F401

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """Resolve command functions lazily through the CLI package (PEP 562)."""
    from abstracts_explorer import cli

    if name not in cli._LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(cli, name)
    globals()[name] = value
    return value


def _command(name: str) -> Any:
    """Return a command callable by name, honouring module-level patches."""
    return getattr(sys.modules[__name__], name)


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

    # CLI flag overrides the READ_ONLY configuration value.
    if getattr(args, "read_only", None):
        config.read_only = True

    # Setup logging based on verbosity
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    if config.read_only and _command_writes_storage(args):
        print(
            "\n❌ This command writes to the paper database or embedding store "
            "and is unavailable in read-only mode.",
            file=sys.stderr,
        )
        print("   Unset READ_ONLY or remove --read-only to allow writes.\n", file=sys.stderr)
        return 1

    if args.command == "download":
        return _command("download_command")(args)
    elif args.command == "create-embeddings":
        return _command("create_embeddings_command")(args)
    elif args.command == "pre-process":
        return _command("pre_process_command")(args)
    elif args.command == "search":
        return _command("search_command")(args)
    elif args.command == "chat":
        return _command("chat_command")(args)
    elif args.command == "web-ui":
        return _command("web_ui_command")(args)
    elif args.command == "clustering":
        if not hasattr(args, "clustering_command") or not args.clustering_command:
            groups["clustering"].print_help()
            return 1
        if args.clustering_command == "run":
            return _command("cluster_embeddings_command")(args)
        elif args.clustering_command == "clear-cache":
            return _command("clear_clustering_cache_command")(args)
        elif args.clustering_command == "pre-generate":
            return _command("pre_generate_clustering_command")(args)
        else:
            groups["clustering"].print_help()
            return 1
    elif args.command == "mcp-server":
        return _command("mcp_server_command")(args)
    elif args.command == "eval":
        if not hasattr(args, "eval_command") or not args.eval_command:
            groups["eval"].print_help()
            return 1
        if args.eval_command == "generate":
            return _command("eval_generate_command")(args)
        elif args.eval_command == "verify":
            return _command("eval_verify_command")(args)
        elif args.eval_command == "run":
            return _command("eval_run_command")(args)
        elif args.eval_command == "results":
            return _command("eval_results_command")(args)
        elif args.eval_command == "clear":
            return _command("eval_clear_command")(args)
        else:
            groups["eval"].print_help()
            return 1
    elif args.command == "check-consistency":
        return _command("consistency_check_command")(args)
    elif args.command == "delete-data":
        return _command("delete_data_command")(args)
    elif args.command == "feedback":
        if not hasattr(args, "feedback_command") or not args.feedback_command:
            groups["feedback"].print_help()
            return 1
        if args.feedback_command == "stats":
            return _command("feedback_stats_command")(args)
        elif args.feedback_command == "list":
            return _command("feedback_list_command")(args)
        elif args.feedback_command == "browse":
            return _command("feedback_browse_command")(args)
        elif args.feedback_command == "clear":
            return _command("feedback_clear_command")(args)
        else:
            groups["feedback"].print_help()
            return 1
    elif args.command == "registry":
        if not hasattr(args, "registry_command") or not args.registry_command:
            groups["registry"].print_help()
            return 1
        if args.registry_command == "upload":
            return _command("registry_upload_command")(args)
        elif args.registry_command == "download":
            return _command("registry_download_command")(args)
        elif args.registry_command == "list":
            return _command("registry_list_command")(args)
        elif args.registry_command == "delete":
            return _command("registry_delete_command")(args)
        else:
            groups["registry"].print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
