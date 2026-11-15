"""
Cluster labeling using TF-IDF keywords and LLM-generated labels.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.tfidf import compute_tfidf_keywords, fit_tfidf, keywords_from_matrix
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.llm_utils import chat_completion_text

logger = logging.getLogger(__name__)


class LabelingMixin:
    """Extract keywords and generate cluster/hierarchy labels."""

    embeddings_manager: Any
    metadatas: Any
    paper_ids: Any
    cluster_labels: Any
    cluster_keywords: Any
    cluster_label_names: Any
    cluster_hierarchy: Any

    def generate_hierarchical_labels(self, use_llm: bool = True, max_keywords: int = 5) -> Dict[int, str]:
        """
        Generate labels for all levels of the hierarchy.

        Uses a tiered approach when use_llm is True:
        - Levels 1-3: simple fallback (concatenation) for fast processing
        - Level 4: full keyword extraction + LLM label generation
        - Level 5+: LLM-based parent label generation from child labels

        Parameters
        ----------
        use_llm : bool, optional
            Whether to use LLM for label generation, by default True
        max_keywords : int, optional
            Maximum number of keywords to use in label generation, by default 5

        Returns
        -------
        Dict[int, str]
            Dictionary mapping node IDs to labels

        Raises
        ------
        ClusteringError
            If hierarchy not available
        """
        if self.cluster_hierarchy is None or "tree" not in self.cluster_hierarchy:
            raise ClusteringError("Cluster hierarchy not available. Use agglomerative clustering.")

        # First generate labels for leaf clusters if not already done
        if self.cluster_label_names is None:
            self.generate_cluster_labels(use_llm=use_llm, max_keywords=max_keywords)

        tree = self.cluster_hierarchy["tree"]
        hierarchical_labels = {}

        # Start with leaf labels (map sample indices to cluster labels)
        n_samples = self.cluster_hierarchy["n_samples"]
        for i in range(n_samples):
            if self.cluster_labels is not None:
                cluster_id = int(self.cluster_labels[i])
                if self.cluster_label_names and cluster_id in self.cluster_label_names:
                    hierarchical_labels[i] = self.cluster_label_names[cluster_id]
                else:
                    hierarchical_labels[i] = f"Sample {i}"

        # Create mapping from paper IDs to indices for LLM label generation
        paper_id_to_idx = {}
        if self.paper_ids:
            paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids)}

        # Generate labels for internal nodes bottom-up with tiered approach
        for level in range(1, tree["max_level"] + 1):
            for node_id, node_info in tree["nodes"].items():
                if node_info["level"] == level:
                    child_labels = [
                        hierarchical_labels.get(child, f"Node {child}") for child in node_info["children"]
                    ]

                    if use_llm:
                        # Tiered approach based on level
                        if level <= 3:
                            # Levels 0-3: Use simple fallback (fast)
                            hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                            logger.debug(f"Level {level} node {node_id}: Using fallback")
                        elif level == 4:
                            # Level 4: Use full cluster label generation with keywords
                            try:
                                if self.embeddings_manager and self.metadatas:
                                    keywords = self._extract_keywords_for_samples(node_info["samples"], max_keywords)
                                    if keywords:
                                        label = self._generate_llm_label_from_keywords(keywords)
                                        hierarchical_labels[node_id] = label
                                        logger.debug(
                                            f"Level {level} node {node_id}: Generated LLM label from keywords {keywords}"
                                        )
                                    else:
                                        hierarchical_labels[node_id] = self._generate_parent_label_fallback(
                                            child_labels
                                        )
                                else:
                                    hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                            except Exception as e:
                                logger.warning(f"Keyword-based label generation failed for node {node_id}: {e}")
                                hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                        else:
                            # Levels 5+: Use LLM-based parent label generation
                            if self.embeddings_manager:
                                try:
                                    # Convert paper IDs to indices for metadata lookup
                                    sample_indices = [
                                        paper_id_to_idx[pid] for pid in node_info["samples"] if pid in paper_id_to_idx
                                    ]
                                    label = self._generate_parent_label_llm(child_labels, sample_indices)
                                    hierarchical_labels[node_id] = label
                                    logger.debug(f"Level {level} node {node_id}: Generated LLM parent label")
                                except Exception as e:
                                    logger.warning(f"LLM label generation failed for node {node_id}: {e}")
                                    hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                            else:
                                hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                    else:
                        # LLM disabled: always use fallback
                        hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)

        return hierarchical_labels

    def _llm_chat_label(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        """
        Shared code path for one-shot LLM cluster-label generation.

        Uses ``config.cluster_label_model`` (default ``alias-fast``) for the
        call.  The back-end chat models are *reasoning* models that emit a
        hidden reasoning block before the short 3-5 word label.  We pass
        ``reasoning_effort="low"`` to suppress that block where the back-end
        supports it (GPT-OSS-120B does; Qwen3.8 ignores the param and always
        reasons ~140-150 tokens).  ``max_tokens`` therefore stays at 200 as a
        safety net: with ``reasoning_effort="low"`` only ~10-20 tokens are
        needed, but the Qwen fallback path still needs the full budget
        (default 200) or ``message.content`` comes back ``None`` with
        ``finish_reason="length"``.  Callers are expected to wrap this in a
        try/except and provide a fallback label.

        Parameters
        ----------
        system_prompt : str
            System message for the LLM.
        user_prompt : str
            User message for the LLM.
        max_tokens : int
            Maximum tokens to generate (default 200).  With
            ``reasoning_effort="low"`` only a handful of tokens are used;
            the budget exists to absorb reasons from models where the param
            is a no-op.

        Returns
        -------
        str
            The model's response, stripped of surrounding whitespace and
            any enclosing single/double quotes.

        Raises
        ------
        AttributeError
            If the OpenAI client is not available on the embeddings manager.
        ValueError
            If the model returns no text content.
        """
        if not hasattr(self.embeddings_manager, "openai_client"):
            raise AttributeError("OpenAI client not available in embeddings manager")

        from abstracts_explorer.config import get_config

        config = get_config()

        content = chat_completion_text(
            client=self.embeddings_manager.openai_client,
            model=config.cluster_label_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=max_tokens,
            # Suppress the hidden reasoning block on GPT-OSS-120B so the
            # ~5-token label fits comfortably.  Qwen3.8 ignores this param
            # (its reasoning is controlled by enable_thinking, which the
            # deployment does not honor) and falls back on max_tokens.
            reasoning_effort="low",
        )

        if not content:
            raise ValueError("LLM returned empty content")
        return content.strip().strip("\"'")

    def _call_llm_for_label(
        self,
        prompt: str,
        fallback: str,
        system_prompt: str = "You are a research paper categorization expert.",
    ) -> str:
        """
        Generate a label with the LLM, falling back to a static label on failure.

        Parameters
        ----------
        prompt : str
            User message containing the labeling instructions.
        fallback : str
            Label to return if the LLM call fails or returns no content.
        system_prompt : str, optional
            System message for the LLM, by default
            ``"You are a research paper categorization expert."``

        Returns
        -------
        str
            LLM-generated label, or ``fallback`` when generation fails.
        """
        try:
            return self._llm_chat_label(system_prompt, prompt)
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return fallback

    def _generate_parent_label_llm(self, child_labels: List[str], sample_indices: List[int]) -> str:
        """
        Generate a parent cluster label by summarizing child labels using LLM.

        Parameters
        ----------
        child_labels : List[str]
            Labels of child clusters
        sample_indices : List[int]
            Indices of samples in this parent cluster

        Returns
        -------
        str
            Generated parent label
        """
        # Get sample titles from the parent cluster
        sample_titles = []
        if self.metadatas and len(sample_indices) > 0:
            sample_size = min(5, len(sample_indices))
            sampled_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
            for idx in sampled_indices:
                title = self.metadatas[idx].get("title", "")
                if title:
                    sample_titles.append(title)

        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles) if sample_titles else "N/A"
        child_labels_str = "\n".join(f"- {label}" for label in set(child_labels))

        prompt = f"""Given a parent cluster that contains the following sub-clusters:

{child_labels_str}

Sample paper titles from this parent cluster:
{sample_titles_str}

Generate a concise, descriptive label (3-5 words) that captures the overarching theme of this parent cluster.
The label should generalize the themes of the child clusters.
Only respond with the label, nothing else. Do not add formatting."""
        logger.debug(f"Generating LLM parent label with prompt: {prompt}")

        return self._call_llm_for_label(
            prompt,
            self._generate_parent_label_fallback(child_labels),
            system_prompt=(
                "You are a research paper categorization expert. "
                "Generate concise labels that generalize child cluster themes."
            ),
        )

    def _generate_parent_label_fallback(self, child_labels: List[str]) -> str:
        """
        Generate a parent label by combining child labels (fallback method).

        Parameters
        ----------
        child_labels : List[str]
            Labels of child clusters

        Returns
        -------
        str
            Generated parent label
        """
        # Split each label by "&", flatten, deduplicate, and rejoin
        labels = []
        for label in child_labels[:3]:
            labels += label.split(" & ")
        # Deduplicate while preserving order
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
            List of paper IDs
        max_keywords : int, optional
            Maximum number of keywords to extract

        Returns
        -------
        List[str]
            List of extracted keywords
        """
        if not self.metadatas or not self.paper_ids:
            return []

        # Find indices for the given paper IDs
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
            List of keywords

        Returns
        -------
        str
            Generated label
        """
        keywords_str = ", ".join(keywords[:5])
        prompt = f"""Given these keywords from academic papers: {keywords_str}

Generate a concise, descriptive label (3-5 words) that captures the main theme.
Only respond with the label, nothing else. Do not add formatting."""

        return self._call_llm_for_label(
            prompt,
            ", ".join(keywords[:3]),
            system_prompt="You are a research paper categorization expert.",
        )

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
            Number of top keywords to extract per cluster, by default 10
        min_df : int, optional
            Minimum document frequency for a term to be considered, by default 2

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of keywords

        Raises
        ------
        ClusteringError
            If clustering has not been performed or metadata is missing

        Examples
        --------
        >>> cm = ClusteringManager(em)
        >>> cm.load_embeddings()
        >>> cm.cluster(n_clusters=5)
        >>> keywords = cm.extract_cluster_keywords(n_keywords=10)
        >>> print(f"Cluster 0 keywords: {keywords[0]}")
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if self.metadatas is None:
            raise ClusteringError("No metadata available.")

        try:
            # Get unique cluster labels (excluding noise)
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
    ) -> Dict[int, str]:
        """
        Generate descriptive labels for clusters.

        This method can either use an LLM to generate meaningful labels based on
        cluster keywords and representative papers, or simply concatenate keywords.

        Parameters
        ----------
        use_llm : bool, optional
            Whether to use LLM for label generation, by default True
        max_keywords : int, optional
            Maximum number of keywords to use in label generation, by default 5

        Returns
        -------
        Dict[int, str]
            Dictionary mapping cluster labels to descriptive names

        Raises
        ------
        ClusteringError
            If clustering or keyword extraction has not been performed

        Examples
        --------
        >>> cm = ClusteringManager(em)
        >>> cm.load_embeddings()
        >>> cm.cluster(n_clusters=5)
        >>> cm.extract_cluster_keywords()
        >>> labels = cm.generate_cluster_labels(use_llm=True)
        >>> print(f"Cluster 0 label: {labels[0]}")
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")

        # Extract keywords if not already done
        if self.cluster_keywords is None:
            logger.info("Extracting cluster keywords first...")
            self.extract_cluster_keywords()

        try:
            self.cluster_label_names = {}
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            for cluster_id in cluster_ids:
                keywords = (self.cluster_keywords or {}).get(cluster_id, [])[:max_keywords]

                if not keywords:
                    self.cluster_label_names[cluster_id] = f"Cluster {cluster_id}"
                    continue

                if use_llm and self.embeddings_manager:
                    try:
                        # Generate label using LLM
                        label = self._generate_llm_label(cluster_id, keywords)
                        self.cluster_label_names[cluster_id] = label
                    except Exception as e:
                        logger.warning(f"LLM label generation failed for cluster {cluster_id}: {e}")
                        # Fallback to keyword-based label
                        self.cluster_label_names[cluster_id] = ", ".join(keywords[:3])
                else:
                    # Use keyword-based label
                    self.cluster_label_names[cluster_id] = ", ".join(keywords[:3])

            logger.info(f"Generated labels for {len(self.cluster_label_names)} clusters")
            return self.cluster_label_names

        except Exception as e:
            raise ClusteringError(f"Failed to generate cluster labels: {str(e)}") from e

    def _generate_llm_label(self, cluster_id: int, keywords: List[str]) -> str:
        """
        Generate a cluster label using LLM.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier
        keywords : List[str]
            List of keywords for the cluster

        Returns
        -------
        str
            Generated label
        """
        # Get a few representative paper titles from the cluster
        assert self.cluster_labels is not None
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        sample_size = min(5, len(cluster_indices))

        # Use replacement if there are fewer papers than sample size
        replace = len(cluster_indices) < sample_size
        sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=replace)

        sample_titles = []
        if self.metadatas:
            for idx in sample_indices:
                title = self.metadatas[idx].get("title", "")
                if title:
                    sample_titles.append(title)

        # Construct prompt for LLM
        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles)
        prompt = f"""Given a cluster of research papers with the following characteristics:

Top keywords: {', '.join(keywords)}

Sample paper titles:
{sample_titles_str}

        Generate a concise, descriptive label (3-5 words) that captures the main theme of this cluster.
Only respond with the label, nothing else. Do not add formatting."""

        return self._call_llm_for_label(
            prompt,
            ", ".join(keywords[:3]),
            system_prompt=(
                "You are a research paper categorization expert. "
                "Generate concise, descriptive labels for clusters of papers."
            ),
        )
