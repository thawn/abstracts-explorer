"""
Backfill missing LLM cluster descriptions from cached clustering results.
"""

import logging
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.labels import combine_label_description, split_label_description
from abstracts_explorer.clustering.representation import ProjectLLMRepresentation
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


def _normalize_int_map(values: Mapping[Any, Any] | None) -> Dict[int, Any]:
    """Normalize cached JSON keys to integers."""
    if not values:
        return {}
    return {int(key): value for key, value in values.items()}


def _has_description(value: Any) -> bool:
    """Return whether a description contains non-whitespace text."""
    return bool(str(value or "").strip())


def _set_map_value(mapping: Dict[Any, Any], cluster_id: int, value: Any) -> None:
    """Set a cached mapping value while preserving the existing key style."""
    if cluster_id in mapping:
        mapping[cluster_id] = value
    elif str(cluster_id) in mapping:
        mapping[str(cluster_id)] = value
    elif mapping and all(isinstance(key, str) for key in mapping):
        mapping[str(cluster_id)] = value
    else:
        mapping[cluster_id] = value


def find_flat_clusters_missing_descriptions(results: Mapping[str, Any]) -> List[int]:
    """
    Return flat cluster IDs without an LLM-generated description.

    Parameters
    ----------
    results : mapping
        Cached clustering results.

    Returns
    -------
    list of int
        Sorted flat cluster IDs whose descriptions are missing or blank.
    """
    labels = _normalize_int_map(results.get("cluster_labels"))
    descriptions = _normalize_int_map(results.get("cluster_descriptions"))
    statistics = results.get("statistics") or {}
    sizes = _normalize_int_map(statistics.get("cluster_sizes"))

    cluster_ids = {cluster_id for cluster_id in sizes if cluster_id >= 0}
    cluster_ids.update(cluster_id for cluster_id in labels if cluster_id >= 0)
    return sorted(cluster_id for cluster_id in cluster_ids if not _has_description(descriptions.get(cluster_id)))


def find_hierarchy_clusters_missing_descriptions(results: Mapping[str, Any]) -> Dict[int, frozenset[int]]:
    """
    Return merged hierarchy nodes without an LLM-generated description.

    Parameters
    ----------
    results : mapping
        Cached clustering results.

    Returns
    -------
    dict
        Mapping of hierarchy node ID to its member flat topic IDs.

    Raises
    ------
    ClusteringError
        If the cached results do not contain a hierarchy tree.
    """
    hierarchy = results.get("hierarchy") or {}
    if "tree" not in hierarchy:
        raise ClusteringError(
            "Cached clustering results are missing their hierarchy tree; "
            "regenerate them with 'abstracts-explorer clustering cache populate --force'."
        )

    missing: Dict[int, frozenset[int]] = {}
    for node in hierarchy.get("tree") or []:
        members = frozenset(int(topic) for topic in node.get("topics") or [] if int(topic) >= 0)
        if len(members) <= 1:
            continue
        if _has_description(node.get("description")):
            continue
        missing.setdefault(int(node.get("id", -1)), members)
    return missing


def _hierarchy_labels_by_members(results: Mapping[str, Any]) -> Dict[frozenset[int], str]:
    """Collect successful hierarchy labels keyed by their member flat topics."""
    return _tree_labels_by_members(results)


def _load_documents(
    results: Mapping[str, Any],
    database: Any,
) -> tuple[List[str], List[int]]:
    """Rebuild documents and cached topic assignments for a clustering result."""
    points = list(results.get("points") or [])
    if not points:
        raise ClusteringError("Cannot backfill cluster labels: cached results contain no points")

    paper_ids: List[str] = []
    for point in points:
        paper_id = point.get("id")
        if paper_id is None:
            raise ClusteringError("Cannot backfill cluster labels: a cached point has no paper ID")
        paper_ids.append(str(paper_id))

    try:
        papers = database.get_paper_summaries_by_uids(paper_ids)
    except Exception as exc:
        raise ClusteringError(f"Failed to load papers for cluster-label backfill: {exc}") from exc

    papers_by_uid = {str(paper.get("uid")): paper for paper in papers}
    missing = [paper_id for paper_id in paper_ids if paper_id not in papers_by_uid]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ClusteringError(
            "Cannot backfill cluster labels: missing paper metadata for "
            f"{len(missing)} cached point(s): {preview}{suffix}"
        )

    documents: List[str] = []
    for paper_id in paper_ids:
        try:
            documents.append(EmbeddingsManager.embedding_text_from_paper(papers_by_uid[paper_id]))
        except Exception as exc:
            raise ClusteringError(f"Cannot create embedding text for paper {paper_id}: {exc}") from exc

    topics = [int(point.get("cluster", -1)) for point in points]
    return documents, topics


def _tree_labels_by_members(results: Mapping[str, Any]) -> Dict[frozenset[int], str]:
    """Collect tree labels/descriptions keyed by their member flat topics."""
    labels: Dict[frozenset[int], str] = {}
    hierarchy = results.get("hierarchy") or {}
    for node in hierarchy.get("tree") or []:
        members = frozenset(int(topic) for topic in node.get("topics") or [] if int(topic) >= 0)
        if len(members) <= 1:
            continue
        if not _has_description(node.get("description")):
            continue
        labels[members] = combine_label_description(node.get("label", ""), node.get("description"))
    return labels


class CachedLabelRepresentation(ProjectLLMRepresentation):
    """Reuse existing labels and generate only missing cluster descriptions."""

    def __init__(
        self,
        embeddings_manager: Any,
        *,
        flat_labels: Mapping[int, str],
        flat_descriptions: Mapping[int, str],
        hierarchy_labels: Mapping[frozenset[int], str],
        flat_topics: Sequence[int],
        target_flat_ids: set[int] | None = None,
        target_hierarchy_members: set[frozenset[int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(embeddings_manager, **kwargs)
        self._flat_labels = dict(flat_labels)
        self._flat_descriptions = dict(flat_descriptions)
        self._hierarchy_labels = dict(hierarchy_labels)
        self._flat_topics = list(flat_topics)
        self._target_flat_ids = None if target_flat_ids is None else set(target_flat_ids)
        self._target_hierarchy_members = None if target_hierarchy_members is None else set(target_hierarchy_members)

    def _resolve_cached_label(self, topic: Any, documents: Any) -> str | None:
        if int(topic) == -1:
            return combine_label_description("Outliers", "")

        try:
            selection = documents.loc[documents["Topic"] == topic, :]
            if "ID" in selection.columns:
                document_ids = [int(index) for index in selection["ID"].tolist()]
            else:
                document_ids = [int(index) for index in selection.index.tolist()]
        except Exception:
            return None

        members = frozenset(
            self._flat_topics[document_id]
            for document_id in document_ids
            if 0 <= document_id < len(self._flat_topics)
        )
        if not members:
            return combine_label_description("Merged cluster", "")
        if members == frozenset({-1}):
            return combine_label_description("Outliers", "")

        if len(members) == 1:
            cluster_id = next(iter(members))
            description = self._flat_descriptions.get(cluster_id)
            if _has_description(description):
                label = self._flat_labels.get(cluster_id) or f"Topic {cluster_id}"
                return combine_label_description(label, description)
            if self._target_flat_ids is None or cluster_id in self._target_flat_ids:
                return None
            label = self._flat_labels.get(cluster_id) or f"Topic {cluster_id}"
            return combine_label_description(label, "")

        cached = self._hierarchy_labels.get(members)
        if cached is not None:
            return cached
        if self._target_hierarchy_members is None or members in self._target_hierarchy_members:
            return None
        return combine_label_description("Merged cluster", "")


def _new_flat_labels(
    topic_model: Any,
    missing: Sequence[int],
) -> Dict[int, tuple[str, str]]:
    """Return successfully generated flat labels for the supplied cluster IDs."""
    labels: Dict[int, tuple[str, str]] = {}
    for cluster_id in missing:
        try:
            entries = topic_model.get_topic(cluster_id)
        except Exception as exc:
            logger.warning("Failed to read backfilled label for topic %s: %s", cluster_id, exc)
            continue
        if not entries:
            continue
        label, description = split_label_description(entries[0][0])
        if _has_description(description):
            labels[cluster_id] = (label or f"Topic {cluster_id}", description)
    return labels


def _hierarchy_labels_from_topics(hierarchical_topics: Any) -> Dict[frozenset[int], str]:
    """Map BERTopic hierarchy member sets to encoded labels/descriptions."""
    labels: Dict[frozenset[int], str] = {}
    if hierarchical_topics is None:
        return labels
    for row in hierarchical_topics.itertuples(index=False):
        members = frozenset(int(topic) for topic in row.Topics)
        label, description = split_label_description(row.Parent_Name)
        if _has_description(description):
            labels[members] = combine_label_description(label, description)
    return labels


def _apply_flat_labels(
    results: Dict[str, Any],
    labels: Mapping[int, tuple[str, str]],
) -> int:
    """Apply generated flat labels and return the number of updates."""
    if not labels:
        return 0
    result_labels = results.setdefault("cluster_labels", {})
    descriptions = results.setdefault("cluster_descriptions", {})
    for cluster_id, (label, description) in labels.items():
        _set_map_value(result_labels, cluster_id, label)
        _set_map_value(descriptions, cluster_id, description)
    return len(labels)


def _apply_hierarchy_labels(
    results: Dict[str, Any],
    labels: Mapping[frozenset[int], str],
) -> int:
    """Apply generated hierarchy labels and return the number of updates."""
    hierarchy = results.get("hierarchy") or {}
    if "tree" not in hierarchy:
        raise ClusteringError(
            "Cached clustering results are missing their hierarchy tree; "
            "regenerate them with 'abstracts-explorer clustering cache populate --force'."
        )

    updated = 0
    nodes_by_members = {
        frozenset(int(topic) for topic in node.get("topics") or [] if int(topic) >= 0): node
        for node in hierarchy.get("tree") or []
    }
    for members, encoded in labels.items():
        node = nodes_by_members.get(members)
        if node is None:
            continue
        if _has_description(node.get("description")):
            continue
        label, description = split_label_description(encoded)
        if not _has_description(description):
            continue
        node["label"] = label or node.get("label", "")
        node["description"] = description
        updated += 1
    return updated


def _distance_function(matrix: Any) -> Any:
    """Return the distance function used during clustering."""
    return np.clip(1 - cosine_similarity(matrix), 0, None)


def backfill_missing_labels(
    results: Dict[str, Any],
    embeddings_manager: Any,
    database: Any,
    *,
    show_progress: bool = False,
    doc_length: int = 500,
    nr_docs: int = 4,
    diversity: float | None = 0.1,
) -> Dict[str, int]:
    """
    Backfill missing flat and hierarchy cluster descriptions.

    Successful labels are reused through a caching representation model;
    only clusters without a description are sent to the LLM.

    Parameters
    ----------
    results : dict
        Cached clustering results. Updated in place on success.
    embeddings_manager : EmbeddingsManager
        Manager exposing the OpenAI-compatible LLM client.
    database : DatabaseManager
        Database used to rebuild paper documents.
    show_progress : bool, optional
        Show tqdm progress bars for the LLM labeling batches (default False).
    doc_length : int, optional
        Maximum characters per representative document (default 500).
    nr_docs : int, optional
        Number of representative documents per cluster (default 4).
    diversity : float or None, optional
        Representative-document diversity (default 0.1).

    Returns
    -------
    dict
        Counts for ``missing``, ``flat_updated``, ``hierarchy_updated``, and
        ``failed``.
    """
    flat_missing = find_flat_clusters_missing_descriptions(results)
    hierarchy_missing = find_hierarchy_clusters_missing_descriptions(results)
    report = {
        "missing": len(flat_missing) + len(hierarchy_missing),
        "flat_updated": 0,
        "hierarchy_updated": 0,
        "failed": 0,
    }
    if report["missing"] == 0:
        return report

    documents, topics = _load_documents(results, database)
    flat_labels = _normalize_int_map(results.get("cluster_labels"))
    flat_descriptions = _normalize_int_map(results.get("cluster_descriptions"))

    representation = CachedLabelRepresentation(
        embeddings_manager,
        flat_labels=flat_labels,
        flat_descriptions=flat_descriptions,
        hierarchy_labels=_hierarchy_labels_by_members(results),
        flat_topics=topics,
        target_flat_ids=set(flat_missing),
        target_hierarchy_members=set(hierarchy_missing.values()),
        nr_docs=nr_docs,
        diversity=diversity,
        doc_length=doc_length,
        show_progress=show_progress,
    )

    try:
        topic_model = BERTopic(
            embedding_model=None,
            representation_model=representation,
            verbose=False,
        )
        topic_model.topics_ = list(topics)
        topic_model.update_topics(documents, topics=list(topics), representation_model=representation)
        new_flat = _new_flat_labels(topic_model, flat_missing)

        new_hierarchy: Dict[frozenset[int], str] = {}
        if hierarchy_missing:
            hierarchical_topics = topic_model.hierarchical_topics(documents, distance_function=_distance_function)
            new_hierarchy = _hierarchy_labels_from_topics(hierarchical_topics)
    except ClusteringError:
        raise
    except Exception as exc:
        raise ClusteringError(f"Failed to backfill cluster labels: {exc}") from exc

    report["flat_updated"] = _apply_flat_labels(results, new_flat)
    report["hierarchy_updated"] = _apply_hierarchy_labels(results, new_hierarchy)
    report["failed"] = len(find_flat_clusters_missing_descriptions(results)) + len(
        find_hierarchy_clusters_missing_descriptions(results)
    )

    logger.info(
        "Backfilled cluster labels: %d flat, %d hierarchy; %d still missing",
        report["flat_updated"],
        report["hierarchy_updated"],
        report["failed"],
    )
    return report
