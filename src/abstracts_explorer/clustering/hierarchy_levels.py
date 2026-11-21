"""
BERTopic hierarchy level adapter.

BERTopic 0.17.4 does not expose the slider level mappings outside its Plotly
visualization. This module consumes the DataFrame returned by
``topic_model.hierarchical_topics()`` and mirrors only the level-selection and
mapping loops from ``bertopic.plotting._hierarchical_documents``. Keep this
adapter in sync with the pinned BERTopic fork.
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from abstracts_explorer.clustering.labels import split_label_description

logger = logging.getLogger(__name__)


def compute_level_max_distances(
    hierarchical_topics: Any,
    nr_levels: int,
    level_scale: str = "linear",
) -> List[float]:
    """
    Compute the maximum distance for each hierarchy level.

    The computation mirrors BERTopic's ``visualize_hierarchical_documents``:
    distances are split into ``nr_levels`` chunks and the last distance of
    each chunk is used as that level's maximum.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()`` containing a
        ``Distance`` column in BERTopic's returned order.
    nr_levels : int
        Requested number of levels.
    level_scale : str, optional
        ``"linear"`` (default) or ``"log"``/``"logarithmic"``.

    Returns
    -------
    list of float
        Ascending maximum distances, one per level.
    """
    distances = hierarchical_topics["Distance"].to_list()
    if not distances or nr_levels <= 0:
        return []

    nr_levels = min(int(nr_levels), len(distances))

    if level_scale in ("log", "logarithmic"):
        if len(distances) < 2:
            return [float(distances[0])]
        log_indices = (
            np.round(
                np.logspace(
                    start=math.log(1, 10),
                    stop=math.log(len(distances) - 1, 10),
                    num=nr_levels,
                )
            )
            .astype(int)
            .tolist()
        )
        log_indices.reverse()
        return [float(distances[index]) for index in log_indices]

    if level_scale not in ("lin", "linear"):
        raise ValueError("level_scale needs to be one of 'log' or 'linear'")

    max_distances = [
        float(distances[indices[-1]]) for indices in np.array_split(range(len(hierarchical_topics)), nr_levels)
    ]
    return list(reversed(max_distances))


def build_level_mapping(
    hierarchical_topics: Any,
    max_distance: float,
    flat_topic_ids: Sequence[int],
) -> Dict[int, int]:
    """
    Build the flat-topic to level-cluster mapping for one level.

    This mirrors the mapping loop in BERTopic's
    ``visualize_hierarchical_documents``.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.
    max_distance : float
        Maximum hierarchy distance for the level.
    flat_topic_ids : sequence of int
        Flat topic IDs to include in the mapping.

    Returns
    -------
    dict
        Mapping from flat topic ID to the cluster ID at the level. The
        mapping is resolved transitively so every value is a final cluster ID.
    """
    mapping: Dict[int, int] = {int(topic): int(topic) for topic in flat_topic_ids}

    if hierarchical_topics is None or len(hierarchical_topics) == 0:
        return mapping

    selection = hierarchical_topics.loc[hierarchical_topics["Distance"] <= max_distance, :].copy()
    if selection.empty:
        return mapping

    selection["Parent_ID"] = selection["Parent_ID"].astype(int)
    selection = selection.sort_values("Parent_ID")

    for row in selection.itertuples(index=False):
        parent_id = int(row.Parent_ID)
        for topic in row.Topics:
            mapping[int(topic)] = parent_id

    changed = True
    while changed:
        changed = False
        for key, value in list(mapping.items()):
            resolved = mapping.get(value, value)
            if resolved != value:
                mapping[key] = resolved
                changed = True

    return mapping


def build_level_mappings(
    hierarchical_topics: Any,
    flat_topic_ids: Sequence[int],
    nr_levels: int,
    level_scale: str = "linear",
) -> List[Dict[str, Any]]:
    """
    Build all hierarchy level mappings.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.
    flat_topic_ids : sequence of int
        Flat topic IDs, including ``-1`` when outliers exist.
    nr_levels : int
        Requested number of levels.
    level_scale : str, optional
        ``"linear"`` (default) or ``"log"``/``"logarithmic"``.

    Returns
    -------
    list of dict
        One entry per level with ``level``, ``max_distance``, and ``mapping``
        keys. Returns an empty list when no hierarchy is available.
    """
    max_distances = compute_level_max_distances(hierarchical_topics, nr_levels, level_scale)
    levels: List[Dict[str, Any]] = []
    for index, max_distance in enumerate(max_distances):
        levels.append(
            {
                "level": index,
                "max_distance": max_distance,
                "mapping": build_level_mapping(hierarchical_topics, max_distance, flat_topic_ids),
            }
        )
    return levels


def _parent_names(hierarchical_topics: Any) -> Mapping[int, str]:
    """Return parent ID to raw parent-name mappings from hierarchy rows."""
    if hierarchical_topics is None or len(hierarchical_topics) == 0:
        return {}
    return {
        int(parent_id): str(parent_name)
        for parent_id, parent_name in zip(
            hierarchical_topics["Parent_ID"].astype(int),
            hierarchical_topics["Parent_Name"],
        )
    }


def level_parent_labels(hierarchical_topics: Any) -> Mapping[int, str]:
    """
    Return parent ID to short parent label mappings from BERTopic hierarchy rows.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.

    Returns
    -------
    Mapping[int, str]
        Parent ID to short label mapping.
    """
    return {
        parent_id: split_label_description(parent_name)[0]
        for parent_id, parent_name in _parent_names(hierarchical_topics).items()
    }


def level_parent_descriptions(hierarchical_topics: Any) -> Mapping[int, str]:
    """
    Return parent ID to descriptive-sentence mappings from BERTopic hierarchy rows.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.

    Returns
    -------
    Mapping[int, str]
        Parent ID to description mapping. Descriptions are empty when the
        hierarchy was built without LLM label descriptions.
    """
    return {
        parent_id: split_label_description(parent_name)[1]
        for parent_id, parent_name in _parent_names(hierarchical_topics).items()
    }
