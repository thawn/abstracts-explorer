"""
BERTopic clustering manager.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTER_N_COMPONENTS,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_LINKAGE,
    DEFAULT_METRIC,
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_NR_LEVELS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_LEVEL_SCALE,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.hierarchy_levels import level_parent_labels
from abstracts_explorer.clustering.loading import LoadingMixin
from abstracts_explorer.clustering.representation import ProjectLLMRepresentation
from abstracts_explorer.clustering.results import build_clustering_results, export_results_to_json
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


class ClusteringManager(LoadingMixin):
    """
    Load embeddings and fit a BERTopic model.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance to load embeddings from.
    database : DatabaseManager
        Database manager for fetching paper metadata.
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        database: DatabaseManager,
    ) -> None:
        self.embeddings_manager = embeddings_manager
        self.database = database
        self.embeddings: Optional[np.ndarray] = None
        self.paper_ids: Optional[List[str]] = None
        self.metadatas: Optional[List[Dict[str, Any]]] = None
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.topics: Optional[np.ndarray] = None
        self.probabilities: Optional[Any] = None
        self.topic_model: Optional[Any] = None
        self.results: Optional[Dict[str, Any]] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_label_names: Optional[Dict[int, str]] = None
        self.cluster_keywords: Optional[Dict[int, List[str]]] = None
        self.cluster_hierarchy: Optional[Dict[str, Any]] = None

    @staticmethod
    def _make_umap(n_components: int) -> Any:
        """Create a UMAP reducer with the fixed project settings."""
        from umap.umap_ import UMAP

        return UMAP(
            n_neighbors=DEFAULT_N_NEIGHBORS,
            n_components=n_components,
            min_dist=DEFAULT_MIN_DIST,
            metric="cosine",
            random_state=DEFAULT_RANDOM_STATE,
        )

    @staticmethod
    def _make_clusterer() -> Any:
        """Create the fixed Agglomerative clusterer."""
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(
            n_clusters=None,  # type: ignore[arg-type]
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
            linkage=DEFAULT_LINKAGE,
            metric=DEFAULT_METRIC,
            compute_full_tree=True,  # type: ignore[arg-type]
        )

    def _build_documents(self) -> List[str]:
        """Build one document per paper from metadata."""
        if not self.metadatas:
            return []
        return [EmbeddingsManager.embedding_text_from_paper(metadata) for metadata in self.metadatas]

    def _extract_flat_keywords(self, docs: List[str], topics: np.ndarray) -> Dict[int, List[str]]:
        """Extract base c-TF-IDF keywords for each flat topic."""
        import pandas as pd

        if self.topic_model is None:
            return {}

        try:
            documents = pd.DataFrame({"Document": docs, "Topic": topics})
            words = list(self.topic_model.vectorizer_model.get_feature_names_out())
            keywords_by_topic = self.topic_model._extract_words_per_topic(
                words,
                documents,
                self.topic_model.c_tf_idf_,
                fine_tune_representation=False,
                calculate_aspects=False,
            )
            keywords: Dict[int, List[str]] = {}
            for topic_id, entries in keywords_by_topic.items():
                topic_int = int(topic_id)
                if topic_int < 0:
                    continue
                keywords[topic_int] = [str(word) for word, _ in entries[:10] if word]
            return keywords
        except Exception as exc:
            logger.warning("Failed to extract flat c-TF-IDF keywords: %s", exc)
            return {}

    def _extract_flat_labels(
        self,
        topics: np.ndarray,
        flat_keywords: Dict[int, List[str]],
        use_llm_labels: bool,
    ) -> Dict[int, str]:
        """Return one label per flat topic."""
        if self.topic_model is None:
            return {}

        labels: Dict[int, str] = {}
        for topic_id in sorted(int(topic) for topic in np.unique(topics) if int(topic) >= 0):
            label = ""
            if use_llm_labels:
                try:
                    entries = self.topic_model.get_topic(topic_id)
                    if entries:
                        label = str(entries[0][0]).strip()
                except Exception as exc:
                    logger.warning("Failed to read LLM label for topic %s: %s", topic_id, exc)

            label = " ".join(label.split()).strip("\"'")
            if label.lower().startswith("topic:"):
                label = label.split(":", 1)[1].strip()
            if not label:
                keywords = flat_keywords.get(topic_id, [])
                label = " & ".join(keywords[:3]) if keywords else f"Topic {topic_id}"
            labels[topic_id] = label[:120]
        return labels

    def fit(
        self,
        nr_levels: int = DEFAULT_NR_LEVELS,
        level_scale: str = DEFAULT_LEVEL_SCALE,
        use_llm_labels: bool = True,
        show_progress: bool = False,
        cluster_n_components: int = DEFAULT_CLUSTER_N_COMPONENTS,
    ) -> Dict[str, Any]:
        """
        Fit BERTopic and build the clustering result payload.

        Parameters
        ----------
        nr_levels : int, optional
            Number of hierarchy levels (default 10).
        level_scale : str, optional
            Level scaling mode (default ``"linear"``).
        use_llm_labels : bool, optional
            Whether flat and hierarchy topics use LLM labels (default True).
        show_progress : bool, optional
            Enable BERTopic verbose output (default False).
        cluster_n_components : int, optional
            UMAP components used before clustering (default 5).

        Returns
        -------
        dict
            Clustering results with points, statistics, flat fields, and
            hierarchy levels.
        """
        if self.embeddings is None or self.paper_ids is None or self.metadatas is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        from bertopic import BERTopic

        docs = self._build_documents()
        if not docs:
            raise ClusteringError("No documents available for clustering")

        representation_model = (
            ProjectLLMRepresentation(self.embeddings_manager, show_progress=show_progress) if use_llm_labels else None
        )
        topic_model = BERTopic(
            verbose=show_progress,
            umap_model=self._make_umap(cluster_n_components),
            hdbscan_model=self._make_clusterer(),
            embedding_model=None,
            representation_model=representation_model,  # type: ignore[arg-type]
        )

        logger.info("Fitting BERTopic on %d documents...", len(docs))
        topics, probabilities = topic_model.fit_transform(docs, embeddings=self.embeddings)
        self.topic_model = topic_model
        self.topics = np.asarray(topics, dtype=int)
        self.probabilities = probabilities
        topics_list = [int(topic) for topic in self.topics]

        logger.info("Extracting flat c-TF-IDF keywords...")
        flat_keywords = self._extract_flat_keywords(docs, self.topics)
        flat_labels = self._extract_flat_labels(self.topics, flat_keywords, use_llm_labels)

        if use_llm_labels:
            logger.info("Labeling hierarchy topics with LLM (BERTopic merge progress bar)...")
        else:
            logger.info("Building hierarchical topics...")
        from sklearn.metrics.pairwise import cosine_similarity

        def distance_function(matrix):
            return np.clip(1 - cosine_similarity(matrix), 0, None)

        hierarchical_topics = topic_model.hierarchical_topics(docs, distance_function=distance_function)
        parent_labels = level_parent_labels(hierarchical_topics)

        logger.info("Reducing embeddings to 2D with UMAP...")
        reduced_embeddings = self._make_umap(2).fit_transform(self.embeddings)
        self.reduced_embeddings = reduced_embeddings

        results = build_clustering_results(
            paper_ids=self.paper_ids,
            reduced_embeddings=reduced_embeddings,
            metadatas=self.metadatas,
            topics=topics_list,
            flat_labels=flat_labels,
            flat_keywords=flat_keywords,
            hierarchical_topics=hierarchical_topics,
            parent_labels=parent_labels,
            nr_levels=nr_levels,
            level_scale=level_scale,
        )

        self.results = results
        self.cluster_labels = self.topics
        self.cluster_label_names = flat_labels
        self.cluster_keywords = flat_keywords
        self.cluster_hierarchy = results["hierarchy"]
        return results

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Return statistics for the fitted clustering.

        Returns
        -------
        dict
            Cluster statistics.

        Raises
        ------
        ClusteringError
            If clustering has not been performed.
        """
        if self.results is None:
            if self.cluster_labels is None:
                raise ClusteringError("No clustering performed. Call fit() first.")
            from abstracts_explorer.clustering.results import build_statistics

            return build_statistics([int(topic) for topic in self.cluster_labels])
        return self.results["statistics"]

    def get_clustering_results(self) -> Dict[str, Any]:
        """
        Return the complete clustering results.

        Returns
        -------
        dict
            Clustering results.

        Raises
        ------
        ClusteringError
            If clustering has not been performed.
        """
        if self.results is None:
            raise ClusteringError("No clustering performed. Call fit() first.")
        return self.results

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Export clustering results to a JSON file.

        Parameters
        ----------
        output_path : str or Path
            Destination path.
        """
        export_results_to_json(self.get_clustering_results(), output_path)
