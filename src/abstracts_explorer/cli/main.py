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

    parser, _groups = build_parser(config)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # CLI flag overrides the READ_ONLY configuration value.
    if getattr(args, "read_only", None):
        config.read_only = True

    # Setup logging based on verbosity
    setup_logging(args.verbose)

    handler = getattr(args, "_handler", None)
    if handler is None:
        selected_group = getattr(args, "_selected_group", None)
        if selected_group is not None:
            selected_group.print_help()
        else:
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

    return _command(handler)(args)


if __name__ == "__main__":
    sys.exit(main())
