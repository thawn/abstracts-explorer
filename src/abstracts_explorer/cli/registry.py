"""
``registry`` commands

"""

import argparse
import logging
import sys
from typing import Optional


from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager


from abstracts_explorer.cli._common import (
    _confirm,
    _print_header,
    _resolve_conference_arg,
    _resolve_registry_auth,
    _unexpected_error,
)

logger = logging.getLogger(__name__)


def registry_upload_command(args: argparse.Namespace) -> int:
    """
    Upload data for a conference (and optionally a specific year) to an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - conference: Conference name
        - year: Conference year (optional; all years if omitted)
        - tag: Optional custom tag

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    repository, token = _resolve_registry_auth(args)
    # Resolve conference name to canonical form; treat None/"all" as "all"
    raw_conf = getattr(args, "conference", None)
    if raw_conf and raw_conf.lower() != "all":
        conference = _resolve_conference_arg(raw_conf) or raw_conf
    else:
        conference = "all"

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    if not token:
        print(
            "❌ Authentication token not specified. Use --token or set GITHUB_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    only_tag = getattr(args, "only_tag", False)

    _print_header(
        "Registry Re-tag" if only_tag else "Registry Upload",
        f"Repository: {repository}",
        f"Conference: {conference}",
        f"Year: {args.year if args.year else 'all'}",
        f"Tag: {args.tag}" if args.tag else None,
        "Only tag (no data upload)" if only_tag else None,
    )

    if only_tag:
        if conference == "all":
            print(
                "❌ --only-tag requires a specific --conference (not 'all').",
                file=sys.stderr,
            )
            return 1
        if args.tag:
            print("❌ --only-tag cannot be combined with --tag.", file=sys.stderr)
            return 1
    elif conference == "all" and (args.year is not None or args.tag):
        print(
            "❌ --year and --tag cannot be combined with --conference all.",
            file=sys.stderr,
        )
        return 1

    try:
        client = RegistryClient(repository=repository, token=token)

        if only_tag:
            summary = client.retag_conference(
                conference=conference,
                years=[args.year] if args.year is not None else None,
                progress_callback=lambda msg: print(f"  {msg}"),
            )
            print("\n✅ Re-tag complete!")
            print(f"  📅 Years:      {summary.get('years', [])}")
            print(f"  🏷️  Tag:        {summary.get('tag', '')}")
            return 0

        if conference == "all":
            summaries = client.upload_all(
                progress_callback=lambda msg: print(f"  {msg}"),
            )
            print(f"\n✅ Upload complete! Uploaded {len(summaries)} conference(s).")
            for s in summaries:
                print(
                    f"  📦 {s.get('conference', '')}: {s.get('paper_count', 0)} papers, "
                    f"{s.get('embedding_count', 0)} embeddings, "
                    f"{s.get('clustering_cache_count', 0)} cache entries (tag: {s.get('tag', '')})"
                )
        else:
            summary = client.upload(
                conference=conference,
                year=args.year,
                tag=args.tag,
                progress_callback=lambda msg: print(f"  {msg}"),
            )
            print("\n✅ Upload complete!")
            print(f"  📄 Papers:     {summary.get('paper_count', 0)}")
            print(f"  🧮 Embeddings: {summary.get('embedding_count', 0)}")
            print(f"  📦 Cache:      {summary.get('clustering_cache_count', 0)}")
            print(f"  📅 Years:      {summary.get('years', [])}")
            print(f"  🏷️  Tag:        {summary.get('tag', '')}")

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e, log_message="Upload failed")


def registry_update_summary_command(args: argparse.Namespace) -> int:
    """
    Rebuild a conference-level registry tag from existing per-year artifacts.

    Only a small config blob and the summary manifest are uploaded; existing
    per-year data layers are reused by digest, so no data is re-uploaded.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - conference: Conference name (required)
        - embedding_model: Optional embedding model override

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    repository, token = _resolve_registry_auth(args)

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    if not token:
        print(
            "❌ Authentication token not specified. Use --token or set GITHUB_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    raw_conf = getattr(args, "conference", None)
    if not raw_conf or raw_conf.lower() == "all":
        print("❌ --conference is required (a specific conference).", file=sys.stderr)
        return 1
    conference = _resolve_conference_arg(raw_conf) or raw_conf

    _print_header(
        "Registry Update Summary",
        f"Repository: {repository}",
        f"Conference: {conference}",
        f"Embedding model: {args.embedding_model}" if args.embedding_model else None,
    )

    try:
        client = RegistryClient(repository=repository, token=token)
        summary = client.update_conference_summary(
            conference=conference,
            embedding_model=args.embedding_model,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        print("\n✅ Summary updated!")
        print(f"  📄 Papers:     {summary.get('paper_count', 0)}")
        print(f"  🧮 Embeddings: {summary.get('embedding_count', 0)}")
        print(f"  📅 Years:      {summary.get('years', [])}")
        print(f"  🏷️  Tag:        {summary.get('tag', '')}")
        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e, log_message="Update summary failed")


def _registry_download_has_existing_data(conference: str, year: Optional[int]) -> bool:
    """
    Check whether the local database contains data for a registry pull scope.

    Parameters
    ----------
    conference : str
        Conference name to check, or ``"all"`` to check for any local data.
    year : int or None
        Conference year to check.  When ``None``, any year for the conference
        counts as existing data.

    Returns
    -------
    bool
        ``True`` if local data matching the scope exists.  On database errors,
        ``True`` is returned so the caller still asks before replacing data.
    """
    try:
        with DatabaseManager() as db:
            db.create_tables()
            conference_years = db.get_conference_years_from_db()
    except Exception as e:
        logger.warning("Could not inspect local data before registry pull: %s", e)
        return True

    if conference == "all":
        return bool(conference_years)

    for local_conference, years in conference_years.items():
        if local_conference.casefold() == conference.casefold():
            return bool(years) if year is None else year in years
    return False


def registry_download_command(args: argparse.Namespace) -> int:
    """
    Download data for a conference (and optionally a specific year) from an OCI-compatible container registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - conference: Conference name
        - year: Conference year (optional; all years if omitted)
        - tag: Optional custom tag
        - yes: Skip confirmation prompt
        - ignore_embedding_model_mismatch: Proceed despite embedding model mismatch

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import (
        EmbeddingModelMismatchError,
        RegistryClient,
        RegistryError,
    )

    repository, token = _resolve_registry_auth(args)
    # Resolve conference name to canonical form; treat None/"all" as "all"
    raw_conf = getattr(args, "conference", None)
    if raw_conf and raw_conf.lower() != "all":
        conference = _resolve_conference_arg(raw_conf) or raw_conf
    else:
        conference = "all"

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    year_display = str(args.year) if args.year else "all"
    _print_header(
        "Registry Download",
        f"Repository: {repository}",
        f"Conference: {conference}",
        f"Year: {year_display}",
        f"Tag: {args.tag}" if args.tag else None,
    )

    if conference == "all" and (args.year is not None or args.tag):
        print(
            "❌ --year and --tag cannot be combined with --conference all.",
            file=sys.stderr,
        )
        return 1

    if not args.yes and _registry_download_has_existing_data(conference, args.year):
        scope = f"{conference}/{year_display}"
        if conference == "all":
            scope = "all conferences"
        print(f"\n⚠️  Warning: This will replace existing data for {scope}!")
        if not _confirm("Continue?"):
            print("Aborted.")
            return 1

    try:
        client = RegistryClient(repository=repository, token=token)

        embedding_model = getattr(args, "embedding_model", None) or get_config().embedding_model
        ignore_embedding_model_mismatch = getattr(args, "ignore_embedding_model_mismatch", False)

        try:
            if conference == "all":
                summaries = client.download_all(
                    progress_callback=lambda msg: print(f"  {msg}"),
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                )

                print(f"\n✅ Download complete! Downloaded {len(summaries)} artifact(s).")
                for s in summaries:
                    print(
                        f"  📦 {s.get('conference', '')}: {s.get('paper_count', 0)} papers, "
                        f"{s.get('embedding_count', 0)} embeddings, "
                        f"{s.get('clustering_cache_count', 0)} cache entries"
                    )
            else:
                summary = client.download(
                    conference=conference,
                    year=args.year,
                    tag=args.tag,
                    embedding_model=embedding_model,
                    progress_callback=lambda msg: print(f"  {msg}"),
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                )

                print("\n✅ Download complete!")
                print(f"  📄 Papers:     {summary.get('paper_count', 0)}")
                print(f"  🧮 Embeddings: {summary.get('embedding_count', 0)}")
                print(f"  📦 Cache:      {summary.get('clustering_cache_count', 0)}")
                print(f"  📅 Years:      {summary.get('years', [])}")

                metadata = summary.get("metadata", {})
                if metadata:
                    print(f"\n  ℹ️  Artifact version: {metadata.get('version', 'unknown')}")

        except EmbeddingModelMismatchError as mismatch:
            print(
                f"\n❌ Embedding model mismatch:\n"
                f"  Configured model: '{embedding_model}'\n"
                f"  Artifact model:   '{mismatch.remote_model}'\n"
                f"\nIf both names refer to the same model on different backends,\n"
                f"you can use --ignore-embedding-model-mismatch to proceed.\n"
                f"⚠️  Only use this option if you are certain the models are identical!",
                file=sys.stderr,
            )
            return 1

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e, log_message="Download failed")


def registry_list_command(args: argparse.Namespace) -> int:
    """
    List available tags and artifact info in the registry.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - tag: Optional specific tag to inspect

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    repository, token = _resolve_registry_auth(args)

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    try:
        client = RegistryClient(repository=repository, token=token)

        if args.tag:
            # Show details for a specific tag
            info = client.get_artifact_info(args.tag)
            print(f"\nArtifact: {repository}:{args.tag}")
            print("=" * 70)

            annotations = info.get("annotations", {})
            if annotations:
                version = annotations.get("com.abstracts-explorer.version", "unknown")
                conference = annotations.get("com.abstracts-explorer.conference", "unknown")
                years = annotations.get("com.abstracts-explorer.years", "unknown")
                papers = annotations.get("com.abstracts-explorer.paper-count", "?")
                embeddings = annotations.get("com.abstracts-explorer.embedding-count", "?")
                emb_model = annotations.get("com.abstracts-explorer.embedding-model", "")
                print(f"  Version:    {version}")
                print(f"  Conference: {conference}")
                print(f"  Years:      {years}")
                if emb_model:
                    print(f"  Model:      {emb_model}")
                print(f"  📄 Papers:     {papers}")
                print(f"  🧮 Embeddings: {embeddings}")

            for layer in info.get("layers", []):
                size = layer.get("size", 0)
                title = layer.get("annotations", {}).get("org.opencontainers.image.title", "")
                if title:
                    print(f"  📦 {title} ({size / 1024 / 1024:.1f} MB)")
        else:
            # List all tags
            tags = client.list_tags()
            print(f"\nAvailable tags in {repository}:")
            print("=" * 70)
            if tags:
                for tag in sorted(tags):
                    print(f"  • {tag}")
            else:
                print("  (no tags found)")

        return 0

    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e, log_message="List failed")


def registry_delete_command(args: argparse.Namespace) -> int:
    """
    Delete registry package versions below a version and/or untagged versions.

    Uses the GitHub Packages API to list all versions of the container package
    and deletes those whose OCI tag version is strictly below *below_version*.
    When *untagged* is set, untagged (dangling) versions are also deleted after
    each is verified as still untagged and safe to remove.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - repository: OCI repository path
        - token: Authentication token
        - below_version: Optional threshold version (e.g. "0.4.0"); versions older than this are deleted
        - conference: Optional conference filter (only tags for this conference are checked)
        - untagged: When True, also delete verified untagged (dangling) versions
        - dry_run: When True, print what would be deleted without deleting
        - yes: Skip confirmation prompt

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    from abstracts_explorer.registry import RegistryClient, RegistryError

    repository, token = _resolve_registry_auth(args)

    if not repository:
        print(
            "❌ Repository not specified. Use --repository or set REGISTRY_REPOSITORY env var.",
            file=sys.stderr,
        )
        return 1

    if not token:
        print(
            "❌ Authentication token not specified. Use --token or set GITHUB_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    below_version = getattr(args, "below_version", None)
    conference = getattr(args, "conference", None)
    dry_run: bool = getattr(args, "dry_run", False)
    untagged: bool = getattr(args, "untagged", False)

    if below_version is None and not untagged:
        print(
            "❌ Specify --below-version and/or --untagged.",
            file=sys.stderr,
        )
        return 1

    if untagged and conference is not None:
        print(
            "❌ --untagged cannot be combined with --conference "
            "(untagged versions carry no conference information).",
            file=sys.stderr,
        )
        return 1

    _print_header(
        "Registry Delete Old Versions",
        f"Repository: {repository}",
        f"Delete below: {below_version or '(none)'}",
        f"Conference: {conference or 'all'}",
        f"Untagged: {untagged}",
        f"Dry-run: {dry_run}",
    )

    if not dry_run and not getattr(args, "yes", False):
        targets = []
        if below_version is not None:
            targets.append(f"all versions below {below_version}")
        if untagged:
            targets.append("verified untagged versions")
        print(f"\n⚠️  This will permanently delete {' and '.join(targets)} from {repository}.")
        if not _confirm("Type 'yes' to confirm: ", require_yes=True):
            print("Aborted.")
            return 1

    try:
        client = RegistryClient(repository=repository, token=token)
        deleted = client.delete_old_versions(
            below_version=below_version,
            conference=conference,
            dry_run=dry_run,
            progress_callback=lambda msg: print(f"  {msg}"),
            include_untagged=untagged,
        )
        action = "would be deleted" if dry_run else "deleted"
        print(f"\n{'✅' if not dry_run else '🔍'} Done. {len(deleted)} version(s) {action}.")
        if deleted:
            for entry in deleted:
                tags_str = ", ".join(entry.get("tags", []))
                if not tags_str:
                    tags_str = f"(untagged) {entry.get('digest', '')}"
                print(f"  {'[dry-run] ' if dry_run else ''}🗑️  {tags_str}")
        return 0

    except ValueError as e:
        print(f"\n❌ Invalid version: {e}", file=sys.stderr)
        return 1
    except RegistryError as e:
        print(f"\n❌ Registry error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        return _unexpected_error(e, log_message="Delete failed")
