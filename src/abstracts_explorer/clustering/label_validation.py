"""
Detection of fallback/default cluster labels.

Labels produced when LLM labeling fails are stored alongside successful
labels in the clustering cache.  These helpers identify such labels so the
CLI pre-generation commands can regenerate them.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_LABEL_PATTERN = re.compile(r"^(?:Cluster|Sample|Node) \d+$")


def is_default_label(label: str) -> bool:
    """
    Return whether a label is a default fallback name.

    Parameters
    ----------
    label : str
        Cluster or hierarchy node label.

    Returns
    -------
    bool
        ``True`` for labels such as ``"Cluster 3"``, ``"Sample 12"``, or
        ``"Node 42"``.
    """
    return bool(DEFAULT_LABEL_PATTERN.fullmatch(label.strip()))


def join_child_labels(child_labels: Sequence[str]) -> str:
    """
    Build the fallback parent label from child labels.

    This mirrors the historical ``_generate_parent_label_fallback`` logic:
    the first three child labels are split on ``" & "``, flattened,
    deduplicated, and re-joined with ``" & "``.

    Parameters
    ----------
    child_labels : sequence of str
        Labels of the child nodes.

    Returns
    -------
    str
        Concatenated fallback label.
    """
    labels: List[str] = []
    for label in child_labels[:3]:
        labels += label.split(" & ")
    unique_labels = list(dict.fromkeys(labels))
    return " & ".join(unique_labels[:3])


def _normalize_map(values: Mapping[Any, Any]) -> Dict[int, Any]:
    """Normalize integer/string keys from cached JSON to ``int`` keys."""
    return {int(key): value for key, value in values.items()}


def _normalize_keywords(keywords: Optional[Mapping[Any, Sequence[str]]]) -> Dict[int, List[str]]:
    """Normalize integer/string keys from cached JSON to ``int`` keys."""
    if not keywords:
        return {}
    return {int(key): list(value) for key, value in keywords.items()}


def find_default_cluster_labels(
    cluster_labels: Mapping[Any, str],
    cluster_keywords: Optional[Mapping[Any, Sequence[str]]] = None,
) -> List[int]:
    """
    Find flat cluster labels that indicate failed LLM labeling.

    A label is considered failed when it is a default name (``Cluster N``,
    ``Sample N``, ``Node N``) or when it equals the keyword fallback
    ``", ".join(keywords[:3])`` produced by a failed LLM call.

    Parameters
    ----------
    cluster_labels : mapping
        Mapping of cluster ID to label.  Keys may be integers or strings.
    cluster_keywords : mapping, optional
        Mapping of cluster ID to keywords.  When provided, keyword-based
        fallback labels are also detected.

    Returns
    -------
    list of int
        Sorted cluster IDs whose labels should be regenerated.
    """
    labels = _normalize_map(cluster_labels)
    keywords = _normalize_keywords(cluster_keywords)

    failed: List[int] = []
    for cluster_id, label in labels.items():
        if is_default_label(label):
            failed.append(cluster_id)
            continue

        cluster_kws = keywords.get(cluster_id, [])
        if cluster_kws and label.strip() == ", ".join(cluster_kws[:3]):
            failed.append(cluster_id)

    return sorted(failed)


def find_default_hierarchy_labels(
    labels: Mapping[Any, Optional[str]],
    hierarchy: Optional[Mapping[str, Any]] = None,
) -> List[int]:
    """
    Find hierarchy node labels that indicate failed LLM labeling.

    Literal default names are always flagged.  For internal nodes at level 4
    or deeper, labels equal to the child-concatenation fallback are also
    flagged.  Lower levels use that concatenation by design and are left
    untouched.

    Parameters
    ----------
    labels : mapping
        Mapping of node ID to label.  Keys may be integers or strings.
    hierarchy : mapping, optional
        Cluster hierarchy containing ``tree.nodes``.

    Returns
    -------
    list of int
        Sorted node IDs whose labels should be regenerated.
    """
    normalized_labels = _normalize_map(labels)
    if not hierarchy:
        return sorted(
            node_id for node_id, label in normalized_labels.items() if label is not None and is_default_label(label)
        )

    tree = hierarchy.get("tree") or {}
    raw_nodes = tree.get("nodes") or {}
    nodes = _normalize_map(raw_nodes)

    failed = set()
    for node_id, label in normalized_labels.items():
        if label is None:
            continue
        if is_default_label(label):
            failed.add(node_id)
            continue

        node = nodes.get(node_id)
        if not node or int(node.get("level", 0)) < 4:
            continue

        child_labels = [
            normalized_labels.get(int(child)) or f"Node {int(child)}" for child in node.get("children", [])
        ]
        if child_labels and label.strip() == join_child_labels(child_labels):
            failed.add(node_id)

    return sorted(failed)
