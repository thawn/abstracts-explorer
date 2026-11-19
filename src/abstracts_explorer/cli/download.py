"""
``download`` command

"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugins import (
    DownloaderPlugin,
    LightweightPaper,
    get_failed_downloads_path,
    get_plugin,
    get_all_plugins,
    list_plugins,
    list_plugin_names,
)

logger = logging.getLogger(__name__)


def _download_single(
    plugin: DownloaderPlugin,
    year: int,
    output_path: Path,
    force: bool,
    args: argparse.Namespace,
    json_path: Optional[Path] = None,
) -> List[LightweightPaper]:
    """
    Download papers for a single plugin and year.

    Parameters
    ----------
    plugin : DownloaderPlugin
        The plugin instance to use for downloading
    year : int
        Conference year to download
    output_path : Path
        Base output path for intermediate JSON files
    force : bool
        Whether to force re-download
    args : argparse.Namespace
        Original CLI arguments (for plugin-specific options)

    Returns
    -------
    list of LightweightPaper
        Downloaded papers

    Raises
    ------
    Exception
        If the download fails
    """
    kwargs: Dict[str, Any] = {}

    max_workers = getattr(args, "max_workers", None)
    if plugin.plugin_name in {"ml4ps", "agu"} and max_workers is not None:
        kwargs["max_workers"] = max_workers

    if plugin.plugin_name == "agu":
        request_delay = getattr(args, "request_delay", None)
        if request_delay is not None:
            kwargs["request_delay"] = request_delay
        if getattr(args, "no_resume", False):
            kwargs["resume"] = False

    if plugin.plugin_name == "chi":
        input_file = getattr(args, "input_file", None)
        if input_file:
            kwargs["input_path"] = input_file

    target_path = json_path or (output_path.parent / f"{plugin.plugin_name}_{year}.json")
    return plugin.download(year=year, output_path=str(target_path), force_download=force, **kwargs)


def _report_failed_downloads(json_path: Path) -> None:
    """Notify the user about recorded failed URLs and automatic retries."""
    failed_path = Path(get_failed_downloads_path(str(json_path)))
    if not failed_path.exists():
        return
    try:
        data = json.loads(failed_path.read_text(encoding="utf-8"))
        failures = data.get("failed", []) if isinstance(data, dict) else []
        count = len(failures) if isinstance(failures, list) else 0
        label = f"{count} page(s)" if count else "Some pages"
    except (OSError, json.JSONDecodeError):
        label = "Some pages"
    print(f"⚠️  {label} failed; failed URLs saved to {failed_path}", file=sys.stderr)
    print("   They will be retried automatically on the next run.", file=sys.stderr)
    print("   Delete this file to skip retrying them.", file=sys.stderr)


def download_command(
    args: argparse.Namespace,
    on_unit_complete: Optional[Callable[[str, int, int], None]] = None,
) -> int:
    """
    Download conference data and create database.

    Without additional arguments, downloads all available conferences
    for all supported years.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - conference: Name of the downloader plugin to use (None for all)
        - year: Year of conference (None for all supported years)
        - output: Path for intermediate JSON files
        - force: Whether to force re-download
        - list_plugins: Whether to list available plugins
    on_unit_complete : callable, optional
        Called after each successfully downloaded ``(conference, year)`` unit
        has been committed to the database.  Receives the canonical conference
        name, the year, and the number of downloaded papers.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    # Import plugins to register them

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
    plugin_name = getattr(args, "conference", None)

    # Determine which plugins to use
    if plugin_name:
        # Specific plugin requested – look up case-insensitively (plugin names are lowercase)
        plugin = get_plugin(plugin_name.lower())
        if not plugin:
            print(f"❌ Error: Plugin '{plugin_name}' not found", file=sys.stderr)
            print(f"\nAvailable plugins: {', '.join(list_plugin_names())}", file=sys.stderr)
            print("\nUse --list-plugins to see details", file=sys.stderr)
            return 1
        plugins_to_download = [plugin]
    else:
        # No plugin specified: download all plugins that do not require manual input
        plugins_to_download = [
            plugin for plugin in get_all_plugins() if getattr(plugin, "requires_manual_input", False) is not True
        ]

    total_papers = 0
    errors = []

    for plugin in plugins_to_download:
        # Determine years to download
        year = getattr(args, "year", None)
        if year is not None:
            years = [year]
        else:
            years = sorted(plugin.supported_years) if plugin.supported_years else []

        if not years:
            print(f"⚠️  No supported years for {plugin.plugin_name}, skipping.")
            continue

        print(f"Using plugin: {plugin.plugin_description}")
        print(f"Downloading {plugin.plugin_name} for year(s): {', '.join(map(str, years))}...")
        print("=" * 70)

        for yr in years:
            try:
                json_path = output_path.parent / f"{plugin.plugin_name}_{yr}.json"
                papers = _download_single(plugin, yr, output_path, args.force, args)
                print(f"✅ Downloaded {len(papers):,} papers for {plugin.plugin_name} {yr}")
                _report_failed_downloads(json_path)

                with DatabaseManager() as db:
                    db.create_tables()
                    count = db.add_papers(papers)
                    print(f"✅ Loaded {count:,} papers into database")

                total_papers += len(papers)

                if on_unit_complete is not None and papers:
                    unit_conference = papers[0].conference or plugin.plugin_name
                    try:
                        on_unit_complete(unit_conference, yr, len(papers))
                    except Exception as callback_error:
                        logger.warning(
                            "Post-processing callback failed for %s %s: %s",
                            unit_conference,
                            yr,
                            callback_error,
                        )

            except Exception as e:
                msg = f"{plugin.plugin_name} {yr}: {e}"
                errors.append(msg)
                print(f"❌ Error downloading {plugin.plugin_name} {yr}: {e}", file=sys.stderr)
                if args.verbose > 0:
                    traceback.print_exc()

        print()

    config = get_config()
    print(f"💾 Database updated: {config.masked_database_url}")
    print(f"📊 Total papers downloaded: {total_papers:,}")

    if errors:
        print(f"\n⚠️  {len(errors)} error(s) occurred:", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        return 1

    return 0
