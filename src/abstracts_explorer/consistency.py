"""
Cross-database consistency checks.

The paper database, the ChromaDB embedding store, and the clustering cache
are maintained independently.  This module compares them and reports missing,
orphaned, stale, or duplicated data so that operators can repair inconsistencies
before exporting or serving data.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from abstracts_explorer.db.errors import normalize_model_name
from abstracts_explorer.embeddings.hashing import hash_text

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class ConsistencyIssue:
    """
    A single consistency finding.

    Attributes
    ----------
    category : str
        Machine-readable category (e.g. ``"embeddings"``).
    severity : str
        One of ``"error"``, ``"warning"``, or ``"info"``.
    message : str
        Human-readable description of the problem.
    scope : str or None
        Conference/year label the issue belongs to, if any.
    fix : str or None
        Suggested command or action to resolve the issue.
    """

    category: str
    severity: str
    message: str
    scope: Optional[str] = None
    fix: Optional[str] = None


@dataclass
class ConsistencyReport:
    """
    Aggregated result of a consistency check.

    Attributes
    ----------
    issues : list of ConsistencyIssue
        Findings collected during the check.
    fix_cache_ids : list of int
        Clustering cache row IDs that :func:`apply_cache_fixes` may safely
        delete (duplicates and cache rows for scopes without papers).
    scopes_checked : int
        Number of conference/year scopes examined.
    papers_checked : int
        Number of paper rows examined.
    embeddings_checked : int
        Number of embedding records examined.
    """

    issues: List[ConsistencyIssue] = field(default_factory=list)
    fix_cache_ids: List[int] = field(default_factory=list)
    scopes_checked: int = 0
    papers_checked: int = 0
    embeddings_checked: int = 0

    def add(self, issue: ConsistencyIssue) -> None:
        """Append *issue* to the report."""
        self.issues.append(issue)

    @property
    def errors(self) -> List[ConsistencyIssue]:
        """Return all error-severity issues."""
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> List[ConsistencyIssue]:
        """Return all warning-severity issues."""
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def infos(self) -> List[ConsistencyIssue]:
        """Return all info-severity issues."""
        return [issue for issue in self.issues if issue.severity == INFO]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the report to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Report payload with a ``summary`` block and an ``issues`` list.
        """
        return {
            "summary": {
                "scopes_checked": self.scopes_checked,
                "papers_checked": self.papers_checked,
                "embeddings_checked": self.embeddings_checked,
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.infos),
                "fix_cache_ids": list(self.fix_cache_ids),
            },
            "issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity,
                    "message": issue.message,
                    "scope": issue.scope,
                    "fix": issue.fix,
                }
                for issue in self.issues
            ],
        }


def _scope_label(conference: Optional[str], year: Optional[int]) -> str:
    """Return a stable ``Conference Year`` label."""
    return f"{conference} {year}"


def _collection_name(collection: Any) -> Optional[str]:
    """Return the collection name from a Chroma collection or string."""
    if isinstance(collection, str):
        return collection
    return getattr(collection, "name", None)


def _load_embedding_records(collection: Any) -> Dict[str, Optional[str]]:
    """
    Load every embedding ID and its stored document hash from *collection*.

    Parameters
    ----------
    collection : chromadb.Collection
        Collection to read from.

    Returns
    -------
    dict
        Mapping of paper UID to its stored content hash (or ``None``).
    """
    records: Dict[str, Optional[str]] = {}
    batch = collection.get(include=["documents"])
    ids = batch.get("ids") or []
    documents = batch.get("documents") or []
    for index, uid in enumerate(ids):
        records[str(uid)] = documents[index] if index < len(documents) else None
    return records


def _check_embeddings(
    database: Any,
    report: ConsistencyReport,
    records: Dict[str, Optional[str]],
    scopes: List[tuple[str, int]],
    scope_uids: Dict[tuple[str, int], List[str]],
    all_paper_uids: set[str],
) -> None:
    """Compare paper rows with stored embedding records."""
    for conference, year in scopes:
        label = _scope_label(conference, year)
        uids = scope_uids.get((conference, year), [])
        report.papers_checked += len(uids)

        summaries = {str(item["uid"]): item for item in database.get_paper_summaries_by_uids(uids)}
        missing = [uid for uid in uids if uid not in records]
        stale = []
        for uid in uids:
            if uid not in records:
                continue
            summary = summaries.get(uid)
            if summary is None:
                continue
            text = f"{summary.get('title') or ''}\n\n{summary.get('abstract') or ''}".strip()
            if text and records[uid] != hash_text(text):
                stale.append(uid)

        if missing:
            sample = ", ".join(missing[:5])
            report.add(
                ConsistencyIssue(
                    category="embeddings",
                    severity=ERROR,
                    scope=label,
                    message=f"{len(missing)} missing embedding(s) for paper rows (e.g. {sample})",
                    fix="abstracts-explorer create-embeddings --conference " f"{conference} --year {year}",
                )
            )
        if stale:
            sample = ", ".join(stale[:5])
            report.add(
                ConsistencyIssue(
                    category="embeddings",
                    severity=ERROR,
                    scope=label,
                    message=f"{len(stale)} embedding(s) are stale (content changed; e.g. {sample})",
                    fix="abstracts-explorer create-embeddings --conference " f"{conference} --year {year}",
                )
            )

    orphans = [uid for uid in records if uid not in all_paper_uids]
    if orphans:
        sample = ", ".join(orphans[:5])
        report.add(
            ConsistencyIssue(
                category="embeddings",
                severity=ERROR,
                message=f"{len(orphans)} orphan embedding(s) have no matching paper row (e.g. {sample})",
                fix="Re-create embeddings after verifying the paper database, or remove the orphan rows",
            )
        )


def _check_models(database: Any, report: ConsistencyReport, embedding_model: str) -> None:
    """Compare the configured model with the database metadata."""
    metadata_model = database.get_embedding_model()
    if not metadata_model:
        report.add(
            ConsistencyIssue(
                category="model",
                severity=WARNING,
                message="The embedding model metadata is missing from the paper database",
                fix="abstracts-explorer create-embeddings",
            )
        )
        return
    if normalize_model_name(str(metadata_model)) != normalize_model_name(embedding_model):
        report.add(
            ConsistencyIssue(
                category="model",
                severity=ERROR,
                message=(
                    f"Embedding metadata records '{metadata_model}' but the configured model is "
                    f"'{embedding_model}'"
                ),
                fix="Re-create embeddings with the configured model or fix EMBEDDING_MODEL",
            )
        )


def _check_cache(
    database: Any,
    report: ConsistencyReport,
    entries: List[Dict[str, Any]],
    scopes: List[tuple[str, int]],
    paper_scopes: set[tuple[str, int]],
    paper_counts: Dict[tuple[str, int], int],
    collection_name: Optional[str],
    embedding_model: str,
) -> None:
    """Validate clustering cache coverage, duplication, and payloads."""
    entries_by_scope: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for entry in entries:
        entry_conference = entry.get("conference")
        entry_year = entry.get("year")
        if entry_conference is None or entry_year is None:
            continue
        scope = (str(entry_conference), int(entry_year))
        entries_by_scope.setdefault(scope, []).append(entry)

    for scope in scopes:
        if scope not in entries_by_scope:
            conference, year = scope
            report.add(
                ConsistencyIssue(
                    category="cache-coverage",
                    severity=INFO,
                    scope=_scope_label(*scope),
                    message="No clustering cache entry exists for this paper scope",
                    fix="abstracts-explorer clustering pre-generate " f"--conference {conference} --year {year}",
                )
            )

    for scope, scope_entries in entries_by_scope.items():
        label = _scope_label(*scope)
        known_scope = scope in paper_scopes

        if len(scope_entries) > 1:
            ordered = sorted(scope_entries, key=lambda item: int(item["id"]), reverse=True)
            duplicate_ids = [int(item["id"]) for item in ordered[1:]]
            report.fix_cache_ids.extend(duplicate_ids)
            report.add(
                ConsistencyIssue(
                    category="cache-duplicate",
                    severity=WARNING,
                    scope=label,
                    message=(
                        f"{len(scope_entries)} clustering cache entries exist for this scope; "
                        f"keeping id={ordered[0]['id']}, duplicate ids={duplicate_ids}"
                    ),
                    fix="abstracts-explorer check-consistency --fix",
                )
            )

        if not known_scope:
            report.fix_cache_ids.extend(int(entry["id"]) for entry in scope_entries)
            report.add(
                ConsistencyIssue(
                    category="cache-stale",
                    severity=WARNING,
                    scope=label,
                    message="Clustering cache exists but the paper database has no papers for this scope",
                    fix="abstracts-explorer check-consistency --fix",
                )
            )
            continue

        actual_papers = paper_counts.get(scope, 0)
        paper_uids = {str(uid) for uid in database.get_paper_uids({"conference": scope[0], "year": scope[1]})}

        for entry in scope_entries:
            entry_id = int(entry["id"])
            results = entry.get("results") or {}
            points = results.get("points") or []
            statistics = results.get("statistics") or {}
            total = statistics.get("total_papers")

            if total is not None and len(points) != int(total):
                report.add(
                    ConsistencyIssue(
                        category="cache-integrity",
                        severity=ERROR,
                        scope=label,
                        message=(
                            f"Cache entry {entry_id} has {len(points)} point(s) but reports " f"total_papers={total}"
                        ),
                        fix="abstracts-explorer clustering pre-generate "
                        f"--conference {scope[0]} --year {scope[1]} --force",
                    )
                )

            if total is not None and int(total) != actual_papers:
                report.add(
                    ConsistencyIssue(
                        category="cache-integrity",
                        severity=ERROR,
                        scope=label,
                        message=(
                            f"Cache entry {entry_id} reports total_papers={total} but the "
                            f"database has {actual_papers} paper(s)"
                        ),
                        fix="abstracts-explorer clustering pre-generate "
                        f"--conference {scope[0]} --year {scope[1]} --force",
                    )
                )

            missing_points = [str(point.get("id")) for point in points if str(point.get("id")) not in paper_uids]
            if missing_points:
                sample = ", ".join(missing_points[:5])
                report.add(
                    ConsistencyIssue(
                        category="cache-integrity",
                        severity=ERROR,
                        scope=label,
                        message=(
                            f"Cache entry {entry_id} contains {len(missing_points)} point(s) that do "
                            f"not belong to this scope (e.g. {sample})"
                        ),
                        fix="abstracts-explorer clustering pre-generate "
                        f"--conference {scope[0]} --year {scope[1]} --force",
                    )
                )

            params = entry.get("clustering_params") or {}
            entry_collection = params.get("__collection__") if isinstance(params, dict) else None
            if collection_name and entry_collection and entry_collection != collection_name:
                report.add(
                    ConsistencyIssue(
                        category="cache-collection",
                        severity=WARNING,
                        scope=label,
                        message=(
                            f"Cache entry {entry_id} was computed from collection "
                            f"'{entry_collection}' but the active collection is '{collection_name}'"
                        ),
                        fix="abstracts-explorer clustering pre-generate "
                        f"--conference {scope[0]} --year {scope[1]} --force",
                    )
                )

            entry_model = entry.get("embedding_model")
            if entry_model and normalize_model_name(str(entry_model)) != normalize_model_name(embedding_model):
                report.add(
                    ConsistencyIssue(
                        category="model",
                        severity=WARNING,
                        scope=label,
                        message=(
                            f"Cache entry {entry_id} uses embedding model '{entry_model}' but "
                            f"'{embedding_model}' is configured"
                        ),
                        fix="abstracts-explorer clustering pre-generate "
                        f"--conference {scope[0]} --year {scope[1]} --force",
                    )
                )


def _check_collections(
    embeddings_manager: Any,
    report: ConsistencyReport,
    collection_name: Optional[str],
) -> None:
    """Warn about ChromaDB collections other than the active one."""
    client = getattr(embeddings_manager, "client", None)
    if client is None or collection_name is None:
        return
    try:
        collections = client.list_collections()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not list ChromaDB collections: %s", e)
        return

    for collection in collections:
        name = _collection_name(collection)
        if name and name != collection_name:
            report.add(
                ConsistencyIssue(
                    category="collection",
                    severity=WARNING,
                    message=(
                        f"ChromaDB collection '{name}' is not the active collection "
                        f"'{collection_name}' and is not used by the checker"
                    ),
                    fix="Remove the unused collection if it is no longer needed",
                )
            )


def check_consistency(
    database: Any,
    embeddings_manager: Any,
    *,
    embedding_model: str,
    collection_name: Optional[str] = None,
    conference: Optional[str] = None,
    year: Optional[int] = None,
) -> ConsistencyReport:
    """
    Compare papers, embeddings, and clustering cache for consistency.

    Parameters
    ----------
    database : DatabaseManager
        Connected paper database manager.
    embeddings_manager : EmbeddingsManager
        Embeddings manager with an initialized collection.
    embedding_model : str
        Embedding model the application is configured to use.
    collection_name : str, optional
        Active ChromaDB collection name.  Defaults to the embeddings manager's
        ``collection_name`` attribute.
    conference : str, optional
        Restrict the check to one conference.
    year : int, optional
        Restrict the check to one year.

    Returns
    -------
    ConsistencyReport
        Findings, counters, and cache row IDs eligible for safe fixes.
    """
    if collection_name is None:
        collection_name = getattr(embeddings_manager, "collection_name", None)

    report = ConsistencyReport()

    paper_scopes: set[tuple[str, int]] = set()
    paper_counts: Dict[tuple[str, int], int] = {}
    scope_uids: Dict[tuple[str, int], List[str]] = {}
    for conf, years in database.get_conference_years_from_db().items():
        for yr in years:
            paper_scopes.add((conf, int(yr)))

    scopes = sorted(paper_scopes)
    if conference is not None:
        scopes = [scope for scope in scopes if scope[0] == conference]
    if year is not None:
        scopes = [scope for scope in scopes if scope[1] == year]
    report.scopes_checked = len(scopes)

    for scope in scopes:
        scope_uids[scope] = [str(uid) for uid in database.get_paper_uids({"conference": scope[0], "year": scope[1]})]
        paper_counts[scope] = len(scope_uids[scope])

    records = _load_embedding_records(embeddings_manager.collection)
    report.embeddings_checked = len(records)

    all_paper_uids = {str(uid) for uid in database.get_paper_uids()}
    _check_embeddings(database, report, records, scopes, scope_uids, all_paper_uids)
    _check_models(database, report, embedding_model)

    entries = database.list_clustering_cache_entries()
    if conference is not None:
        entries = [entry for entry in entries if entry.get("conference") == conference]
    if year is not None:
        entries = [entry for entry in entries if entry.get("year") == year]
    _check_cache(
        database,
        report,
        entries,
        scopes,
        paper_scopes,
        paper_counts,
        collection_name,
        embedding_model,
    )
    _check_collections(embeddings_manager, report, collection_name)

    return report


def apply_cache_fixes(database: Any, report: ConsistencyReport) -> int:
    """
    Delete clustering cache rows flagged as safe to remove.

    Only duplicate rows and rows for scopes without papers are deleted.
    Embeddings, paper rows, and valid cache entries are never touched.

    Parameters
    ----------
    database : DatabaseManager
        Connected paper database manager.
    report : ConsistencyReport
        Report produced by :func:`check_consistency`.

    Returns
    -------
    int
        Number of cache entries deleted.
    """
    ids = sorted({int(cache_id) for cache_id in report.fix_cache_ids})
    if not ids:
        return 0
    return database.delete_clustering_cache_ids(ids)
