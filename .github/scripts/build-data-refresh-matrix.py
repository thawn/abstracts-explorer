#!/usr/bin/env python3
"""Build the GitHub Actions matrix for the registry data refresh workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from typing import Any

from packaging.version import Version

from abstracts_explorer.config import get_config
from abstracts_explorer.plugins import get_all_plugins
from abstracts_explorer.registry import RegistryClient, _parse_version_from_tag

# GitHub rejects an empty dynamic matrix, so emit a single inert entry when
# there is nothing to do. The refresh job is skipped via the `has_work` output.
_NO_WORK_ENTRY: dict[str, Any] = {
    "plugin": "none",
    "conference": "none",
    "registry_tags": [],
    "is_new": False,
    "target_year_present": True,
    "needs_update": False,
}


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


def _newest_tag(tags: list[str]) -> str:
    """
    Return the tag with the highest PEP 440 version.

    Parameters
    ----------
    tags : list of str
        Tag strings to compare.

    Returns
    -------
    str
        The tag with the highest parsed version. Tags without a parseable
        version sort before versioned tags; the tag string is used as a
        deterministic tie-breaker.
    """

    def _sort_key(tag: str) -> tuple[int, Version, str]:
        parsed = _parse_version_from_tag(tag)
        return (1 if parsed is not None else 0, parsed or Version("0"), tag)

    return max(tags, key=_sort_key)


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

    conference_tags = [tag for tag in tags if tag.startswith(f"{base}_") and f"_{model}_" in tag]
    if conference_tags:
        return [_newest_tag(conference_tags)]

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

    return [_newest_tag(candidates) for _, candidates in sorted(year_tags.items())]


def _target_year_present(
    client: RegistryClient,
    matching_tags: list[str],
    conference: str,
    year: int,
) -> bool:
    """
    Return whether the registry already contains data for *year* in a conference.

    For a conference-level tag the ``com.abstracts-explorer.years`` manifest
    annotation is consulted; otherwise the presence of a matching year-specific
    tag is used.

    Parameters
    ----------
    client : RegistryClient
        Registry client used to inspect manifests.
    matching_tags : list of str
        Tags already matched for the conference (conference-level or year tags).
    conference : str
        Canonical conference name.
    year : int
        Target conference year.

    Returns
    -------
    bool
        True when the target year is already covered by the registry.
    """
    if not matching_tags:
        return False

    base = _sanitize(conference)
    year_pattern = re.compile(rf"^{re.escape(base)}-(\d{{4}})_")
    conference_tags = [tag for tag in matching_tags if not year_pattern.match(tag)]

    if conference_tags:
        tag = _newest_tag(conference_tags)
        try:
            info = client.get_artifact_info(tag)
        except Exception:
            return False
        annotations = info.get("annotations", {})
        years_value = str(annotations.get("com.abstracts-explorer.years", ""))
        present = {part.strip() for part in years_value.split(",") if part.strip()}
        return str(year) in present

    return any(tag.startswith(f"{base}-{year}_") for tag in matching_tags)


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
    only_new = os.environ.get("ONLY_NEW", "true").strip().lower() != "false"

    repository = os.environ.get("REGISTRY_REPOSITORY") or config.registry_repository
    if not repository:
        print("::error::REGISTRY_REPOSITORY is not configured", file=sys.stderr)
        return 1

    token = os.environ.get("GITHUB_TOKEN") or config.github_token

    try:
        registry_client = RegistryClient(repository=repository, token=token)
        registry_tags = registry_client.list_tags()
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
        matching_tags = _matching_tags(
            registry_tags,
            conference=conference,
            embedding_model=config.embedding_model,
            year=year,
        )
        target_year_present = _target_year_present(
            registry_client,
            matching_tags,
            conference=conference,
            year=year,
        )
        entries.append(
            {
                "plugin": plugin.plugin_name,
                "conference": conference,
                "registry_tags": matching_tags,
                "is_new": not matching_tags,
                "target_year_present": target_year_present,
                "needs_update": (not matching_tags) or (not target_year_present),
            }
        )

    selected = [entry for entry in entries if (not only_new) or entry["needs_update"]]
    has_work = bool(selected)
    matrix = {"include": selected if has_work else [_NO_WORK_ENTRY]}

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(matrix)}\n")
            handle.write(f"year={year}\n")
            handle.write(f"has_work={'true' if has_work else 'false'}\n")
    else:
        print(json.dumps(matrix, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
