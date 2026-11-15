"""
Hierarchical cluster structure and dendrogram coordinates.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class HierarchyMixin:
    """Extract the agglomerative merge hierarchy for visualization."""

    clusterer: Any
    cluster_labels: Any
    paper_ids: Any
    cluster_hierarchy: Any

    def _extract_cluster_hierarchy(self) -> None:
        """
        Extract hierarchical cluster structure from agglomerative clustering.

        This method extracts the dendrogram information from scikit-learn's
        AgglomerativeClustering to build a hierarchy that can be used for
        hierarchical visualization.

        The hierarchy is stored in self.cluster_hierarchy as a dictionary
        mapping cluster IDs to their children and parent information.
        """
        from sklearn.cluster import AgglomerativeClustering  # lazy import to reduce initial load time

        if not isinstance(self.clusterer, AgglomerativeClustering):
            return

        if not hasattr(self.clusterer, "children_"):
            logger.warning("Clusterer does not have children_ attribute, hierarchy not available")
            return

        if self.cluster_labels is None:
            logger.warning("Cluster labels not available, cannot extract hierarchy")
            return

        try:
            n_samples = len(self.cluster_labels)
            children = self.clusterer.children_

            # Build hierarchy dictionary
            # Each merge creates a new cluster node
            merges: List[Dict[str, Any]] = []

            # Each row in children represents a merge
            # children[i] = [left, right] where left and right are indices
            # Indices < n_samples are original samples
            # Indices >= n_samples are merged clusters (index - n_samples gives merge step)
            for i, (left, right) in enumerate(children):
                merge_info: Dict[str, Any] = {
                    "merge_id": i,
                    "left": int(left),
                    "right": int(right),
                    "cluster_id": n_samples + i,  # New cluster ID
                }

                # Add distance if available
                if hasattr(self.clusterer, "distances_"):
                    merge_info["distance"] = float(self.clusterer.distances_[i])

                merges.append(merge_info)

            # Build tree structure with levels
            tree = self._build_hierarchy_tree(n_samples, children)

            # Compute dendrogram coordinates for visualization
            dendrogram = self._compute_dendrogram_coords(n_samples, children)

            self.cluster_hierarchy = {
                "n_samples": n_samples,
                "n_clusters": len(np.unique(self.cluster_labels)),
                "merges": merges,
                "tree": tree,
                "dendrogram": dendrogram,
            }
            logger.info(f"Extracted hierarchy with {len(children)} merges")

        except Exception as e:
            logger.warning(f"Failed to extract cluster hierarchy: {e}")
            self.cluster_hierarchy = None

    def _build_hierarchy_tree(self, n_samples: int, children: np.ndarray) -> Dict[str, Any]:
        """
        Build a tree structure from agglomerative clustering merges.

        Parameters
        ----------
        n_samples : int
            Number of original samples
        children : np.ndarray
            Children array from AgglomerativeClustering

        Returns
        -------
        dict
            Tree structure with nodes and their relationships
        """
        # Build node information
        nodes = {}

        # Leaf nodes (original samples)
        # Store paper IDs instead of indices for proper frontend mapping
        for i in range(n_samples):
            paper_id = self.paper_ids[i] if self.paper_ids else str(i)
            nodes[i] = {
                "node_id": i,
                "is_leaf": True,
                "children": [],
                "samples": [paper_id],  # Use paper ID instead of index
                "level": 0,
            }

        # Internal nodes (merges)
        for i, (left, right) in enumerate(children):
            node_id = n_samples + i
            left_node = nodes[int(left)]
            right_node = nodes[int(right)]

            # Extract samples and level info, casting to proper types
            left_samples = list(left_node["samples"])  # type: ignore
            right_samples = list(right_node["samples"])  # type: ignore
            # Remove duplicates while preserving order using dict.fromkeys()
            all_samples = list(dict.fromkeys(left_samples + right_samples))

            left_level: int = left_node["level"]  # type: ignore
            right_level: int = right_node["level"]  # type: ignore

            nodes[node_id] = {
                "node_id": node_id,
                "is_leaf": False,
                "children": [int(left), int(right)],
                "samples": all_samples,  # Deduplicated list of paper IDs
                "level": max(left_level, right_level) + 1,
            }

        # Root is the last merge
        root_id = n_samples + len(children) - 1

        return {"nodes": {k: v for k, v in nodes.items()}, "root": root_id, "max_level": nodes[root_id]["level"]}

    def _compute_dendrogram_coords(self, n_samples: int, children: np.ndarray) -> Dict[str, Any]:
        """
        Compute dendrogram coordinates for visualization.

        This follows the approach from sklearn's plot_dendrogram example:
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

        Parameters
        ----------
        n_samples : int
            Number of original samples
        children : np.ndarray
            Children array from AgglomerativeClustering (shape: [n_merges, 2])

        Returns
        -------
        dict
            Dictionary containing dendrogram plotting data with keys:
            - icoord: list of x-coordinates for vertical lines
            - dcoord: list of y-coordinates (distances) for vertical lines
            - ivl: list of leaf labels at the bottom
            - leaves: list of original leaf indices
            - color_list: list of colors for each link (all default for now)
        """
        n_merges = len(children)

        # Track counts: how many original samples in each cluster
        counts = np.zeros(n_merges + n_samples, dtype=int)
        counts[:n_samples] = 1  # Leaves have count 1

        # Compute counts for internal nodes
        for i, (left, right) in enumerate(children):
            counts[n_samples + i] = counts[int(left)] + counts[int(right)]

        # Track positions of clusters on the x-axis
        positions = {}

        # Initial x-positions for leaves (spread from 5 to 5*n_samples in steps of 10)
        for i in range(n_samples):
            positions[i] = 5.0 + 10.0 * i

        # Dendrogram coordinates
        icoord = []  # x-coordinates for plotting
        dcoord = []  # y-coordinates (heights/distances)

        # Get distances if available
        if hasattr(self.clusterer, "distances_") and self.clusterer is not None:
            distances = self.clusterer.distances_  # type: ignore
        else:
            # If no distances, use merge order as proxy (each merge adds 1 to height)
            distances = np.arange(1, n_merges + 1, dtype=float)

        # Build dendrogram by processing merges
        for i, (left, right) in enumerate(children):
            left_idx = int(left)
            right_idx = int(right)
            node_idx = n_samples + i

            # Get positions of left and right children
            left_pos = positions[left_idx]
            right_pos = positions[right_idx]

            # Position of this merge (midpoint of children)
            merge_pos = (left_pos + right_pos) / 2.0
            positions[node_idx] = merge_pos

            # Get heights of children (0 for leaves, previous merge distance for internals)
            left_height = 0.0 if left_idx < n_samples else distances[left_idx - n_samples]
            right_height = 0.0 if right_idx < n_samples else distances[right_idx - n_samples]
            merge_height = distances[i]

            # Add line coordinates for the merge
            # Format: [left_x, left_x, right_x, right_x] for x-coords
            #         [left_height, merge_height, merge_height, right_height] for y-coords
            icoord.append([left_pos, left_pos, right_pos, right_pos])
            dcoord.append([left_height, merge_height, merge_height, right_height])

        # Create leaf labels (ordered left to right by x-position)
        leaf_order = sorted(range(n_samples), key=lambda x: positions[x])
        ivl = [str(i) for i in leaf_order]

        return {
            "icoord": icoord,
            "dcoord": dcoord,
            "ivl": ivl,
            "leaves": leaf_order,
            "color_list": ["#808080"] * n_merges,  # Default gray for all links
        }
