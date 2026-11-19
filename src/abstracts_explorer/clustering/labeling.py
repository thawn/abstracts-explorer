"""
Cluster labeling using TF-IDF keywords and LLM-generated labels.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.tfidf import compute_tfidf_keywords, fit_tfidf, keywords_from_matrix
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.llm_utils import chat_completion_text

logger = logging.getLogger(__name__)

_RETRYABLE_LLM_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_TFIDF_LLM_LEVEL = 6


@contextmanager
def _progress_bar(total: int, desc: str, unit: str, enabled: bool) -> Iterator[Optional[Any]]:
    """
    Yield a tqdm progress bar when enabled, or ``None`` otherwise.

    Parameters
    ----------
    total : int
        Number of items to process.
    desc : str
        Progress bar description.
    unit : str
        Unit label for processed items.
    enabled : bool
        Whether progress reporting is enabled.

    Yields
    ------
    tqdm or None
        Configured progress bar, or ``None`` when disabled.
    """
    if not enabled or total <= 0:
        yield None
        return

    with tqdm(total=total, desc=desc, unit=unit) as progress:
        yield progress


def _is_retryable_llm_error(error: Exception) -> bool:
    """
    Return whether a cluster-label LLM error should be retried.

    Parameters
    ----------
    error : Exception
        Exception raised by the LLM client.

    Returns
    -------
    bool
        ``True`` for transient network errors and retryable HTTP statuses.
    """
    if isinstance(error, AttributeError):
        return False

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code is None:
        return True

    try:
        return int(status_code) in _RETRYABLE_LLM_STATUS_CODES
    except (TypeError, ValueError):
        return True


class LabelingMixin:
    """Extract keywords and generate cluster/hierarchy labels."""

    embeddings_manager: Any
    metadatas: Any
    paper_ids: Any
    cluster_labels: Any
    cluster_keywords: Any
    cluster_label_names: Any
    cluster_hierarchy: Any

    def generate_hierarchical_labels(
        self,
        use_llm: bool = True,
        max_keywords: int = 5,
        show_progress: bool = False,
        existing_labels: Optional[Dict[int, str]] = None,
        on_label: Optional[Any] = None,
    ) -> Dict[int, str]:
        """
        Generate labels for all levels of the hierarchy.

        Levels 1-5 use TF-IDF keywords derived from the previous hierarchy
        level and sampled paper titles. Levels 6 and deeper use an LLM with
        those keywords and sample titles.

        Parameters
        ----------
        use_llm : bool, optional
            Whether to use the LLM for levels 6 and deeper, by default True.
            When False, those levels use TF-IDF labels like levels 1-5.
        max_keywords : int, optional
            Maximum number of keywords used for label generation, by default 5.
        show_progress : bool, optional
            Show progress bars for hierarchy labeling, by default False.
        existing_labels : dict, optional
            Labels already generated in a previous run. Present labels are
            skipped.
        on_label : callable, optional
            Called as ``on_label(node_id, label)`` for each newly generated
            label.

        Returns
        -------
        Dict[int, str]
            Dictionary mapping node IDs to labels.

        Raises
        ------
        ClusteringError
            If hierarchy or flat cluster labels are unavailable.
        """
        if self.cluster_hierarchy is None or "tree" not in self.cluster_hierarchy:
            raise ClusteringError("Cluster hierarchy not available. Use agglomerative clustering.")
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if not self.cluster_label_names:
            raise ClusteringError("Flat cluster labels are missing; hierarchy labeling was skipped.")

        tree = self.cluster_hierarchy["tree"]
        nodes: Dict[int, Dict[str, Any]] = {int(k): v for k, v in tree["nodes"].items()}
        hierarchical_labels: Dict[int, str] = {
            int(k): v for k, v in (existing_labels or {}).items() if isinstance(v, str) and v.strip()
        }

        cluster_ids = {int(label) for label in np.unique(self.cluster_labels) if label >= 0}
        missing_flat = sorted(cluster_ids - set(self.cluster_label_names))
        if missing_flat:
            raise ClusteringError(
                f"Flat cluster labels are incomplete (missing {len(missing_flat)} clusters); "
                "hierarchy labeling was skipped."
            )

        def save_label(node_id: int, label: str) -> None:
            hierarchical_labels[node_id] = label
            if on_label is not None:
                on_label(node_id, label)

        paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids or [])}
        id_to_cluster = {
            pid: int(self.cluster_labels[idx])
            for pid, idx in paper_id_to_idx.items()
            if idx < len(self.cluster_labels)
        }

        for node_id, node in nodes.items():
            if int(node.get("level", 0)) != 0 or hierarchical_labels.get(node_id):
                continue
            samples = node.get("samples") or []
            cluster_id = id_to_cluster.get(samples[0]) if samples else None
            label = self.cluster_label_names.get(cluster_id) if cluster_id is not None else None
            if not label:
                raise ClusteringError(f"Missing flat cluster label for leaf node {node_id}.")
            save_label(node_id, label)

        def missing_at_level(level: int) -> List[int]:
            return [
                node_id
                for node_id, node in nodes.items()
                if int(node.get("level", 0)) == level and not hierarchical_labels.get(node_id)
            ]

        tfidf_levels = range(1, min(5, int(tree["max_level"])) + 1)
        tfidf_total = sum(len(missing_at_level(level)) for level in tfidf_levels)
        with _progress_bar(
            tfidf_total,
            "Labeling hierarchy (TF-IDF)",
            "node",
            show_progress,
        ) as progress:
            for level in tfidf_levels:
                generated = self._label_level_with_tfidf(
                    nodes,
                    level,
                    hierarchical_labels,
                    save_label,
                    max_keywords,
                )
                if progress is not None and generated:
                    progress.update(generated)

        deep_levels = range(_TFIDF_LLM_LEVEL, int(tree["max_level"]) + 1)
        deep_total = sum(len(missing_at_level(level)) for level in deep_levels)
        deep_desc = "Labeling hierarchy (LLM)" if use_llm else "Labeling hierarchy (deep)"
        with _progress_bar(deep_total, deep_desc, "node", show_progress) as progress:
            for level in deep_levels:
                if use_llm:
                    generated = self._label_level_with_llm(
                        nodes,
                        level,
                        hierarchical_labels,
                        save_label,
                        paper_id_to_idx,
                        max_keywords,
                    )
                else:
                    generated = self._label_level_with_tfidf(
                        nodes,
                        level,
                        hierarchical_labels,
                        save_label,
                        max_keywords,
                    )
                if progress is not None and generated:
                    progress.update(generated)

        return hierarchical_labels

    def _node_document(self, node: Dict[str, Any], child_labels: List[str], sample_indices: List[int]) -> str:
        """Build the TF-IDF input document for a hierarchy node."""
        labels = list(dict.fromkeys(label for label in child_labels if label))
        titles = self._sample_titles(sample_indices)
        return " ".join(labels + titles)

    def _level_tfidf_keywords(
        self,
        nodes: Dict[int, Dict[str, Any]],
        level: int,
        labels: Dict[int, str],
        max_keywords: int,
    ) -> Dict[int, List[str]]:
        """Compute TF-IDF keywords for missing nodes at one hierarchy level."""
        paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids or [])}
        node_ids: List[int] = []
        documents: List[str] = []
        child_labels_by_node: Dict[int, List[str]] = {}

        for node_id, node in nodes.items():
            if int(node.get("level", 0)) != level or labels.get(node_id):
                continue
            child_labels = [labels.get(int(child), "") for child in node.get("children", [])]
            child_labels_by_node[node_id] = child_labels
            sample_indices = [paper_id_to_idx[pid] for pid in (node.get("samples") or []) if pid in paper_id_to_idx]
            node_ids.append(node_id)
            documents.append(self._node_document(node, child_labels, sample_indices))

        if not node_ids:
            return {}

        keywords_by_node: Dict[int, List[str]] = {}
        valid_positions = [i for i, doc in enumerate(documents) if doc.strip()]
        if not valid_positions:
            return {node_id: [] for node_id in node_ids}

        valid_documents = [documents[i] for i in valid_positions]
        try:
            matrix, feature_names = fit_tfidf(valid_documents, min_df=1)
        except Exception as e:
            logger.warning("Failed to compute hierarchy TF-IDF for level %s: %s", level, e)
            return {node_id: [] for node_id in node_ids}

        for position, doc_index in enumerate(valid_positions):
            keywords = keywords_from_matrix(matrix, feature_names, [position], max_keywords)
            keywords_by_node[node_ids[doc_index]] = keywords

        for node_id in node_ids:
            keywords_by_node.setdefault(node_id, [])
        return keywords_by_node

    def _label_level_with_tfidf(
        self,
        nodes: Dict[int, Dict[str, Any]],
        level: int,
        labels: Dict[int, str],
        save_label: Any,
        max_keywords: int,
    ) -> int:
        """Label missing nodes at one level using TF-IDF keywords."""
        keywords_by_node = self._level_tfidf_keywords(nodes, level, labels, max_keywords)
        generated = 0
        for node_id, keywords in keywords_by_node.items():
            child_labels = [labels.get(int(child), "") for child in nodes[node_id].get("children", [])]
            label = ", ".join(keywords[:3]) if keywords else self._generate_parent_label_fallback(child_labels)
            save_label(node_id, label)
            generated += 1
        return generated

    def _label_level_with_llm(
        self,
        nodes: Dict[int, Dict[str, Any]],
        level: int,
        labels: Dict[int, str],
        save_label: Any,
        paper_id_to_idx: Dict[str, int],
        max_keywords: int,
    ) -> int:
        """Label missing nodes at one level using the LLM."""
        keywords_by_node = self._level_tfidf_keywords(nodes, level, labels, max_keywords)
        generated = 0
        for node_id, keywords in keywords_by_node.items():
            node = nodes[node_id]
            sample_indices = [paper_id_to_idx[pid] for pid in (node.get("samples") or []) if pid in paper_id_to_idx]
            try:
                label = self._generate_parent_label_llm(keywords, sample_indices)
            except Exception as e:
                raise ClusteringError(f"Failed to generate hierarchy label for node {node_id}: {e}") from e
            save_label(node_id, label)
            generated += 1
        return generated

    def _llm_chat_label(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        """
        Make one retried LLM call for cluster-label generation.

        Parameters
        ----------
        system_prompt : str
            System message for the LLM.
        user_prompt : str
            User message for the LLM.
        max_tokens : int
            Maximum tokens to generate (default 200).

        Returns
        -------
        str
            The model's response, stripped of surrounding whitespace and quotes.

        Raises
        ------
        AttributeError
            If the OpenAI client is not available.
        ValueError
            If the model returns no text content after all attempts.
        """
        if not hasattr(self.embeddings_manager, "openai_client"):
            raise AttributeError("OpenAI client not available in embeddings manager")

        from abstracts_explorer.config import get_config

        config = get_config()
        max_attempts = max(1, config.llm_max_attempts)
        retry_base_delay = config.llm_retry_base_delay
        retry_max_delay = config.llm_retry_max_delay

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                content = chat_completion_text(
                    client=self.embeddings_manager.openai_client,
                    model=config.cluster_label_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    timeout=config.llm_request_timeout,
                    reasoning_effort="low",
                )

                if not content:
                    raise ValueError("LLM returned empty content")
                return content.strip().strip("\"'")
            except Exception as error:
                last_error = error
                if attempt >= max_attempts or not _is_retryable_llm_error(error):
                    raise

                delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
                logger.warning(
                    "Cluster-label LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    error,
                    delay,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def _call_llm_for_label(
        self,
        prompt: str,
        system_prompt: str = "You are a research paper categorization expert.",
    ) -> str:
        """
        Generate a label with the LLM, propagating failures.

        Parameters
        ----------
        prompt : str
            User message containing the labeling instructions.
        system_prompt : str, optional
            System message for the LLM.

        Returns
        -------
        str
            LLM-generated label.
        """
        return self._llm_chat_label(system_prompt, prompt)

    def _sample_titles(self, sample_indices: List[int], max_titles: int = 5) -> List[str]:
        """Return up to ``max_titles`` deterministic sample titles."""
        if not self.metadatas or not sample_indices:
            return []

        count = min(max_titles, len(sample_indices))
        if len(sample_indices) <= count:
            selected = sample_indices
        else:
            step = len(sample_indices) / count
            selected = [sample_indices[int(i * step)] for i in range(count)]

        titles: List[str] = []
        for idx in selected:
            title = self.metadatas[idx].get("title", "")
            if title:
                titles.append(title)
        return titles

    def _generate_parent_label_llm(self, keywords: List[str], sample_indices: List[int]) -> str:
        """
        Generate a parent cluster label from keywords and example papers.

        Parameters
        ----------
        keywords : List[str]
            Keywords derived from the previous hierarchy level and examples.
        sample_indices : List[int]
            Indices of samples in this parent cluster.

        Returns
        -------
        str
            Generated parent label.
        """
        sample_titles = self._sample_titles(sample_indices)
        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles) if sample_titles else "N/A"
        keywords_str = ", ".join(keywords[:5]) if keywords else "N/A"

        prompt = f"""Given a parent cluster with the following keywords derived from its sub-clusters:

{keywords_str}

Sample paper titles from this parent cluster:
{sample_titles_str}

Generate a concise, descriptive label (3-5 words) that captures the overarching theme of this parent cluster.
Only respond with the label, nothing else. Do not add formatting."""
        logger.debug(f"Generating LLM parent label with prompt: {prompt}")

        return self._call_llm_for_label(
            prompt,
            system_prompt=(
                "You are a research paper categorization expert. "
                "Generate concise labels that generalize the parent cluster theme."
            ),
        )

    def _generate_parent_label_fallback(self, child_labels: List[str]) -> str:
        """
        Generate a parent label by combining child labels.

        Parameters
        ----------
        child_labels : List[str]
            Labels of child clusters.

        Returns
        -------
        str
            Generated parent label.
        """
        labels: List[str] = []
        for label in child_labels[:3]:
            labels += label.split(" & ")
        unique_labels = list(dict.fromkeys(labels))
        label = " & ".join(unique_labels[:3])
        logger.debug(f"Generated fallback parent label: {label}")
        return label

    def _extract_keywords_for_samples(self, sample_paper_ids: List[str], max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from a set of papers identified by their IDs.

        Parameters
        ----------
        sample_paper_ids : List[str]
            List of paper IDs.
        max_keywords : int, optional
            Maximum number of keywords to extract.

        Returns
        -------
        List[str]
            List of extracted keywords.
        """
        if not self.metadatas or not self.paper_ids:
            return []

        paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids)}
        sample_indices = [paper_id_to_idx[paper_id] for paper_id in sample_paper_ids if paper_id in paper_id_to_idx]

        if not sample_indices:
            return []

        all_docs = [EmbeddingsManager.embedding_text_from_paper(metadata) for metadata in self.metadatas]

        return compute_tfidf_keywords(all_docs, sample_indices, max_keywords, min_df=2)

    def _generate_llm_label_from_keywords(self, keywords: List[str]) -> str:
        """
        Generate a descriptive label from keywords using LLM.

        Parameters
        ----------
        keywords : List[str]
            List of keywords.

        Returns
        -------
        str
            Generated label.
        """
        keywords_str = ", ".join(keywords[:5])
        prompt = f"""Given these keywords from academic papers: {keywords_str}

Generate a concise, descriptive label (3-5 words) that captures the main theme.
Only respond with the label, nothing else. Do not add formatting."""

        return self._call_llm_for_label(prompt)

    def extract_cluster_keywords(
        self,
        n_keywords: int = 10,
        min_df: int = 2,
    ) -> Dict[int, List[str]]:
        """
        Extract distinctive keywords for each cluster using TF-IDF.

        Parameters
        ----------
        n_keywords : int, optional
            Number of top keywords to extract per cluster, by default 10.
        min_df : int, optional
            Minimum document frequency for a term to be considered, by default 2.

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of keywords.

        Raises
        ------
        ClusteringError
            If clustering has not been performed or metadata is missing.
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if self.metadatas is None:
            raise ClusteringError("No metadata available.")

        try:
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            self.cluster_keywords = {}

            all_docs = [EmbeddingsManager.embedding_text_from_paper(metadata) for metadata in self.metadatas]
            matrix, feature_names = fit_tfidf(all_docs, min_df=min_df)

            for cluster_id in cluster_ids:
                cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

                if len(cluster_indices) == 0:
                    logger.warning(f"No documents found for cluster {cluster_id}")
                    self.cluster_keywords[cluster_id] = []
                    continue

                keywords = keywords_from_matrix(matrix, feature_names, cluster_indices.tolist(), n_keywords)
                self.cluster_keywords[cluster_id] = keywords
                logger.debug(f"Cluster {cluster_id}: {len(keywords)} keywords extracted")

            logger.info(f"Extracted keywords for {len(self.cluster_keywords)} clusters")
            return self.cluster_keywords

        except Exception as e:
            raise ClusteringError(f"Failed to extract cluster keywords: {str(e)}") from e

    def generate_cluster_labels(
        self,
        use_llm: bool = True,
        max_keywords: int = 5,
        show_progress: bool = False,
        existing_labels: Optional[Dict[int, str]] = None,
        on_label: Optional[Any] = None,
    ) -> Dict[int, str]:
        """
        Generate descriptive labels for flat clusters.

        LLM failures are non-fatal: the affected label is left missing so a
        later run can retry it. Labels already present in ``existing_labels``
        are skipped.

        Parameters
        ----------
        use_llm : bool, optional
            Whether to use an LLM for label generation, by default True.
        max_keywords : int, optional
            Maximum number of keywords to use, by default 5.
        show_progress : bool, optional
            Show a progress bar while labeling clusters, by default False.
        existing_labels : dict, optional
            Labels generated in a previous run.
        on_label : callable, optional
            Called as ``on_label(cluster_id, label)`` for each new label.

        Returns
        -------
        Dict[int, str]
            Dictionary mapping cluster IDs to labels. Failed clusters are
            absent.

        Raises
        ------
        ClusteringError
            If clustering or keyword extraction is unavailable.
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")

        if self.cluster_keywords is None:
            logger.info("Extracting cluster keywords first...")
            self.extract_cluster_keywords()

        unique_labels = np.unique(self.cluster_labels)
        cluster_ids = [int(label) for label in unique_labels if label >= 0]
        self.cluster_label_names = {
            int(k): v for k, v in (existing_labels or {}).items() if isinstance(v, str) and v.strip()
        }

        missing = [cluster_id for cluster_id in cluster_ids if not self.cluster_label_names.get(cluster_id)]
        failed: List[int] = []
        with _progress_bar(len(missing), "Labeling clusters", "cluster", show_progress) as progress:
            for cluster_id in missing:
                keywords = (self.cluster_keywords or {}).get(cluster_id, [])[:max_keywords]

                if use_llm and self.embeddings_manager:
                    try:
                        label = self._generate_llm_label(cluster_id, keywords)
                    except Exception as e:
                        logger.warning(f"LLM label generation failed for cluster {cluster_id}: {e}")
                        failed.append(cluster_id)
                    else:
                        self.cluster_label_names[cluster_id] = label
                        if on_label is not None:
                            on_label(cluster_id, label)
                elif keywords:
                    label = ", ".join(keywords[:3])
                    self.cluster_label_names[cluster_id] = label
                    if on_label is not None:
                        on_label(cluster_id, label)
                else:
                    logger.warning(f"No keywords available for cluster {cluster_id}; label left missing")
                    failed.append(cluster_id)

                if progress is not None:
                    progress.update(1)

        if failed:
            logger.warning(
                "Flat labeling incomplete: %d of %d clusters failed (%s)",
                len(failed),
                len(cluster_ids),
                ", ".join(str(cluster_id) for cluster_id in failed[:10]),
            )

        logger.info(f"Generated labels for {len(self.cluster_label_names)} clusters")
        return self.cluster_label_names

    def _generate_llm_label(self, cluster_id: int, keywords: List[str]) -> str:
        """
        Generate a cluster label using the LLM.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier.
        keywords : List[str]
            Keywords for the cluster; may be empty, in which case example
            paper titles are used.

        Returns
        -------
        str
            Generated label.
        """
        assert self.cluster_labels is not None
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        sample_titles = self._sample_titles(cluster_indices.tolist())
        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles) if sample_titles else "N/A"
        keywords_str = ", ".join(keywords) if keywords else "N/A"

        prompt = f"""Given a cluster of research papers with the following characteristics:

Top keywords: {keywords_str}

Sample paper titles:
{sample_titles_str}

Generate a concise, descriptive label (3-5 words) that captures the main theme of this cluster.
Only respond with the label, nothing else. Do not add formatting."""

        return self._call_llm_for_label(
            prompt,
            system_prompt=(
                "You are a research paper categorization expert. "
                "Generate concise, descriptive labels for clusters of papers."
            ),
        )
