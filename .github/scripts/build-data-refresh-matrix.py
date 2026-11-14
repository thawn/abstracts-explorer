#!/usr/bin/env python3
"""Build the GitHub Actions matrix for the registry data refresh workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from typing import Any

from abstracts_explorer.config import get_config
from abstracts_explorer.plugins import get_all_plugins
from abstracts_explorer.registry import RegistryClient


def _sanitize(value: str) -> str:
    """
    Sanitize a string for comparison with OCI tag components.

    Parameters
    ----------
    value : str
        String to sanitize.

    Returns
    -------
    str
        Lowercase string containing only ``[a-z0-9._-]``.
    """
    safe = value.lower()
    safe = re.sub(r"[^a-z0-9._-]", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe)
    return safe.strip("-")


def _matching_tags(
    tags: list[str],
    conference: str,
    embedding_model: str,
    year: int,
) -> list[str]:
    """
    Return the newest usable registry tags for a conference.

    Conference-level tags are preferred because they contain every year.
    When no conference-level tag exists, the newest tag for each available
    year up to *year* is returned.

    Parameters
    ----------
    tags : list of str
        Tags available in the registry.
    conference : str
        Canonical conference name.
    embedding_model : str
        Embedding model from the configuration.
    year : int
        Target conference year.

    Returns
    -------
    list of str
        Matching tags. Empty when no compatible tag exists.
    """
    base = _sanitize(conference)
    model = _sanitize(embedding_model)

    conference_tags = [
        tag for tag in tags if tag.startswith(f"{base}_") and f"_{model}_" in tag
    ]
    if conference_tags:
        return [max(conference_tags)]

    year_tags: dict[int, list[str]] = {}
    pattern = re.compile(rf"^{re.escape(base)}-(\d{{4}})_")
    for tag in tags:
        if f"_{model}_" not in tag:
            continue
        match = pattern.match(tag)
        if match is None:
            continue
        tag_year = int(match.group(1))
        if tag_year <= year:
            year_tags.setdefault(tag_year, []).append(tag)

    return [max(candidates) for _, candidates in sorted(year_tags.items())]


def _conference_filter_matches(plugin: Any, value: str) -> bool:
    """
    Return whether a plugin matches a user-supplied conference filter.

    Parameters
    ----------
    plugin : Any
        Registered downloader plugin instance.
    value : str
        User-supplied plugin name, canonical name, or OCI-safe name.

    Returns
    -------
    bool
        True when the plugin matches the filter.
    """
    if not value:
        return True

    normalized = value.lower()
    candidates = {
        plugin.plugin_name.lower(),
        str(getattr(plugin, "conference_name", plugin.plugin_name)).lower(),
        _sanitize(plugin.plugin_name),
        _sanitize(str(getattr(plugin, "conference_name", plugin.plugin_name))),
    }
    return normalized in candidates or _sanitize(normalized) in candidates


def main() -> int:
    """
    Build and emit the refresh workflow matrix.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    config = get_config()

    target_year = os.environ.get("TARGET_YEAR", "").strip()
    year = int(target_year) if target_year else dt.datetime.now(dt.timezone.utc).year
    conference_filter = os.environ.get("CONFERENCE_FILTER", "").strip()

    repository = os.environ.get("REGISTRY_REPOSITORY") or config.registry_repository
    if not repository:
        print("::error::REGISTRY_REPOSITORY is not configured", file=sys.stderr)
        return 1

    token = os.environ.get("GITHUB_TOKEN") or config.github_token

    try:
        registry_tags = RegistryClient(repository=repository, token=token).list_tags()
    except Exception as exc:
        print(f"::error::Could not list registry tags in {repository}: {exc}", file=sys.stderr)
        return 1

    plugins = []
    for plugin in get_all_plugins():
        if year not in plugin.supported_years:
            continue
        if not _conference_filter_matches(plugin, conference_filter):
            continue
        parameters = plugin.get_metadata().get("parameters", {})
        if isinstance(parameters, dict) and "input_path" in parameters:
            print(f"Skipping '{plugin.plugin_name}': it requires a manually supplied input file")
            continue
        plugins.append(plugin)
    plugins.sort(key=lambda plugin: plugin.plugin_name)

    if not plugins:
        print(
            f"::error::No supported conferences found for {year}"
            + (f" matching '{conference_filter}'" if conference_filter else ""),
            file=sys.stderr,
        )
        return 1

    entries = []
    for plugin in plugins:
        conference = str(getattr(plugin, "conference_name", plugin.plugin_name))
        entries.append(
            {
                "plugin": plugin.plugin_name,
                "conference": conference,
                "registry_tags": _matching_tags(
                    registry_tags,
                    conference=conference,
                    embedding_model=config.embedding_model,
                    year=year,
                ),
            }
        )

    matrix = {"include": entries}
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"year={year}\n")
    else:
        print(json.dumps(matrix, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
