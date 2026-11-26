"""
CLI helpers

"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


from abstracts_explorer.config import Config, get_config

logger = logging.getLogger(__name__)


class _HttpRequestLogFilter(logging.Filter):
    """
    Drop HTTP client request logs unless DEBUG verbosity is enabled.

    ``httpx2`` emits one INFO record per HTTP request. Those records are
    useful when debugging the LLM/embedding backend but noisy during normal
    operation, so they are filtered out unless DEBUG is active. The logger's
    level is left untouched so its own warnings and errors still propagate.

    Attributes
    ----------
    allow_info : bool
        When ``True``, INFO records are allowed through; when ``False`` they
        are dropped. Records above INFO always pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.allow_info = False

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Decide whether a log record should be emitted.

        Parameters
        ----------
        record : logging.LogRecord
            The record being evaluated.

        Returns
        -------
        bool
            ``True`` if the record should be emitted, ``False`` otherwise.
        """
        return self.allow_info or record.levelno > logging.INFO


def _command_writes_storage(args: Any) -> bool:
    """
    Return whether the selected command may write to the paper or embedding store.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    bool
        ``True`` when the command is known to perform database/embedding writes.
    """
    command = getattr(args, "command", None)
    if command in {"download", "create-embeddings", "pre-process", "delete-data"}:
        return True
    if command == "clustering":
        return getattr(args, "clustering_command", None) in {"run", "pre-generate", "clear-cache"}
    if command == "eval":
        return getattr(args, "eval_command", None) in {"generate", "verify", "run", "clear"}
    if command == "registry":
        return getattr(args, "registry_command", None) in {"download", "delete"}
    if command == "feedback":
        return getattr(args, "feedback_command", None) == "clear"
    if command == "check-consistency":
        return bool(getattr(args, "fix", False))
    return False


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

    # Gate httpx2 request logs behind DEBUG without altering its log level.
    httpx_logger = logging.getLogger("httpx2")
    for existing in list(httpx_logger.filters):
        if isinstance(existing, _HttpRequestLogFilter):
            httpx_logger.removeFilter(existing)
    request_filter = _HttpRequestLogFilter()
    request_filter.allow_info = level == logging.DEBUG
    httpx_logger.addFilter(request_filter)


def _print_header(title: str, *lines: Optional[str]) -> None:
    """
    Print a command banner: title line, separator, optional field lines,
    separator.

    Parameters
    ----------
    title : str
        Command title printed after the "Abstracts Explorer - " prefix.
    *lines : str or None, optional
        Field lines to print between the separators.  ``None`` entries are
        skipped, which callers use for optional conference/year lines.
    """
    print(f"Abstracts Explorer - {title}")
    print("=" * 70)
    for line in lines:
        if line is not None:
            print(line)
    print("=" * 70)


def _unexpected_error(e: Exception, label: str = "Unexpected error", log_message: Optional[str] = None) -> int:
    """
    Report an unexpected error and return the CLI exit code.

    Parameters
    ----------
    e : Exception
        The exception to report.
    label : str, optional
        Prefix for the error line, by default "Unexpected error".
    log_message : str, optional
        If given, log the full exception via ``logger.exception`` with this
        message instead of printing a traceback.

    Returns
    -------
    int
        Exit code 1.
    """
    print(f"\n❌ {label}: {e}", file=sys.stderr)
    if log_message:
        logger.exception(log_message)
    else:
        traceback.print_exc()
    return 1


def _confirm(prompt: str, *, require_yes: bool = False) -> bool:
    """
    Ask the user to confirm a destructive action.

    Parameters
    ----------
    prompt : str
        The question to display.  Unless *require_yes* is set, a
        ``[y/N]: `` hint is appended automatically.
    require_yes : bool, optional
        Require the full word ``yes`` instead of accepting ``y``, by
        default False.  Set for prompts that already end in ``: ``.

    Returns
    -------
    bool
        True if the user confirmed.  EOF/interrupt counts as "no".
    """
    try:
        answer = input(f"{prompt}{' [y/N]: ' if not require_yes else ''}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if require_yes:
        return answer == "yes"
    return answer in ("y", "yes")


def _resolve_registry_auth(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    """
    Resolve the OCI repository and auth token for a registry command.

    Command-line arguments take precedence over the ``REGISTRY_REPOSITORY``
    and ``GITHUB_TOKEN`` settings.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments containing ``repository`` and ``token``.

    Returns
    -------
    tuple
        ``(repository, token)``.
    """
    config = get_config()
    repository = args.repository or config.registry_repository
    token = args.token or config.github_token
    return repository, token


def add_conference_year_args(parser: argparse.ArgumentParser) -> None:
    """
    Add standard --conference and --year arguments to a parser.

    Creates a unified, case-insensitive option handler for conference and year
    filtering.  All commands that accept these options should use this helper
    so that help text, types, and defaults are consistent across the CLI.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.
    """
    parser.add_argument(
        "--conference",
        type=str,
        default=None,
        help="Conference to use (default: all conferences). Case-insensitive.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year of conference/workshop (default: all available years).",
    )


def _resolve_conference_arg(conference: Optional[str]) -> Optional[str]:
    """
    Resolve a CLI ``--conference`` value to the canonical conference name.

    Opens the local database, calls
    :py:meth:`~abstracts_explorer.database.DatabaseManager.resolve_conference_name`,
    and returns the result.  If *conference* is ``None``, ``None`` is
    returned.  Raises :class:`~abstracts_explorer.database.DatabaseError`
    when the database cannot be reached or the conference is unknown.

    This helper should be called **once** at the entry point of every CLI
    command that accepts a ``--conference`` option.  After the call, all
    subsequent operations within that command receive the exact name as
    stored in the database (or the plugin's canonical name), preventing
    case-mismatch errors.

    Parameters
    ----------
    conference : str or None
        Raw value from ``args.conference``.

    Returns
    -------
    str or None
        Resolved conference name, or ``None`` if *conference* was ``None``.
    """
    if not conference:
        return conference
    from abstracts_explorer.database import DatabaseManager

    with DatabaseManager() as db:
        db.create_tables()
        resolved_conference = db.resolve_conference_name(conference)
    if resolved_conference is not None and resolved_conference != conference:
        print(f"ℹ️  Resolved conference '{conference}' → '{resolved_conference}'")
    return resolved_conference


def _build_embeddings_where_filters(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """
    Build structured paper filters from --conference and --year arguments.

    The conference name in *args* is expected to already be resolved to its
    canonical form (via ``DatabaseManager.resolve_conference_name``).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments that may contain conference and year attributes.

    Returns
    -------
    dict or None
        Structured where filters, or None if no filters were specified.
    """
    where_filters: Dict[str, Any] = {}
    conference = getattr(args, "conference", None)
    year = getattr(args, "year", None)

    if conference:
        where_filters["conference"] = conference
    if year is not None:
        where_filters["year"] = int(year)

    return where_filters or None


def _validate_embeddings_db(config: Config) -> bool:
    """
    Validate that the configured local embeddings database exists.

    HTTP(S) URLs are always considered valid because the database is remote.

    Parameters
    ----------
    config : Config
        Configuration containing the ``embedding_db`` setting.

    Returns
    -------
    bool
        True if the embeddings database is usable, False otherwise.
    """
    if config.embedding_db.startswith("http://") or config.embedding_db.startswith("https://"):
        return True

    embeddings_path = Path(config.embedding_db)
    if not embeddings_path.exists():
        print(f"❌ Error: Embeddings database not found: {embeddings_path}", file=sys.stderr)
        print("\nYou can create embeddings using:", file=sys.stderr)
        print("  abstracts-explorer create-embeddings", file=sys.stderr)
        return False
    return True
