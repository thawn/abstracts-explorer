"""
Project LLM representation model for BERTopic.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bertopic.representation import BaseRepresentation
from tqdm import tqdm

from abstracts_explorer.clustering.labels import combine_label_description, parse_llm_label_response
from abstracts_explorer.clustering.llm import DEFAULT_LABEL_MAX_TOKENS, _status_code, generate_llm_label
from abstracts_explorer.config import get_config

logger = logging.getLogger(__name__)

LLM_LABEL_PROMPT = """You are labeling a cluster of research papers.

Sample papers:
[DOCUMENTS]

Top keywords: [KEYWORDS]

Return exactly two lines:
Label: a concise, descriptive 3-6 word title-case label for a legend
Description: one complete sentence (max 25 words) describing the shared research theme

Do not add quotes, a "topic:" prefix, or any other text."""

LLM_LABEL_SYSTEM_PROMPT = (
    "You are a research paper categorization expert. Generate a short legend label "
    "and one descriptive sentence for each cluster of papers."
)


class ProjectLLMRepresentation(BaseRepresentation):
    """Generate a short topic label and descriptive sentence per cluster."""

    def __init__(
        self,
        embeddings_manager: Any,
        prompt: str = LLM_LABEL_PROMPT,
        system_prompt: str = LLM_LABEL_SYSTEM_PROMPT,
        nr_docs: int = 4,
        diversity: float | None = 0.1,
        doc_length: int = 500,
        show_progress: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize the representation model.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Embeddings manager exposing an OpenAI-compatible client.
        prompt : str, optional
            Prompt template containing ``[DOCUMENTS]`` and ``[KEYWORDS]``.
        system_prompt : str, optional
            System message for the LLM.
        nr_docs : int, optional
            Number of representative documents per topic (default 4).
        diversity : float or None, optional
            Diversity parameter passed to BERTopic's representative-document
            extraction (default 0.1).
        doc_length : int, optional
            Maximum characters per representative document (default 500).
        show_progress : bool, optional
            Show a tqdm progress bar for multi-topic LLM batches (default False).
        max_workers : int or None, optional
            Maximum number of LLM requests to run in parallel (default
            ``None``, which uses the configured
            :attr:`~abstracts_explorer.config.Config.cluster_label_max_workers`).
            A value of ``1`` labels topics sequentially.
        """
        self.embeddings_manager = embeddings_manager
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.show_progress = show_progress
        self.max_workers = max_workers

    def _resolve_cached_label(self, topic: Any, documents: Any) -> str | None:
        """
        Return a previously generated label for *topic* when available.

        Parameters
        ----------
        topic : Any
            Topic identifier passed by BERTopic.
        documents : Any
            Documents DataFrame associated with the topic batch.

        Returns
        -------
        str or None
            Encoded label/description to reuse, or ``None`` to generate a new
            label.
        """
        return None

    def extract_topics(
        self,
        topic_model: Any,
        documents: Any,
        c_tf_idf: Any,
        topics: Mapping[Any, List[Tuple[str, float]]],
    ) -> Mapping[Any, List[Tuple[str, float]]]:
        """
        Generate short labels and descriptive sentences for the supplied topics.

        Failed or empty LLM responses are retried by
        :func:`generate_llm_label`. When retries are exhausted, the label
        falls back to the topic's top keywords so the clustering run can
        continue.

        Parameters
        ----------
        topic_model : BERTopic
            Fitted BERTopic model.
        documents : pandas.DataFrame
            Documents DataFrame used by BERTopic.
        c_tf_idf : scipy.sparse.csr_matrix
            c-TF-IDF matrix for the supplied topics.
        topics : dict
            Base topic representations mapping topic ID to ``(word, score)``
            tuples.

        Returns
        -------
        dict
            Updated topic representations mapping topic ID to one encoded
            label/description tuple.
        """
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf,
            documents,
            topics,
            500,
            self.nr_docs,
            self.diversity,
        )

        updated_topics: Dict[Any, List[Tuple[str, float]]] = {}
        pending: List[Tuple[Any, List[str], str]] = []
        use_progress = self.show_progress and len(repr_docs_mappings) > 1
        progress = (
            tqdm(
                total=len(repr_docs_mappings),
                desc="Labeling clusters (LLM)",
                unit="topic",
            )
            if use_progress
            else None
        )
        try:
            for topic, docs in repr_docs_mappings.items():
                cached_label = self._resolve_cached_label(topic, documents)
                if cached_label is not None:
                    updated_topics[topic] = [(cached_label, 1)]
                    if progress is not None:
                        progress.update(1)
                    continue

                keywords = [word for word, _ in topics[topic][:10]]
                pending.append((topic, keywords, self._build_prompt(keywords, docs)))

            if pending:
                max_workers = self._effective_max_workers(len(pending))
                if max_workers <= 1:
                    for topic, keywords, prompt in pending:
                        updated_topics[topic] = [(self._label_topic(topic, keywords, prompt), 1)]
                        if progress is not None:
                            progress.update(1)
                else:
                    with ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="cluster-label",
                    ) as executor:
                        futures = {
                            executor.submit(self._label_topic, topic, keywords, prompt): topic
                            for topic, keywords, prompt in pending
                        }
                        for future in as_completed(futures):
                            topic = futures[future]
                            updated_topics[topic] = [(future.result(), 1)]
                            if progress is not None:
                                progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return updated_topics

    def _effective_max_workers(self, task_count: int) -> int:
        """Return the number of labeling threads to use for *task_count* topics."""
        configured = self.max_workers
        if configured is None:
            configured = get_config().cluster_label_max_workers
        return max(1, min(int(configured), task_count))

    def _build_prompt(self, keywords: Sequence[str], docs: Sequence[str]) -> str:
        """Build the label prompt from keywords and representative documents."""
        docs_text = "\n".join(f"- {doc[: self.doc_length]}" for doc in docs)
        prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
        return prompt.replace("[DOCUMENTS]", docs_text)

    def _label_topic(self, topic: Any, keywords: Sequence[str], prompt: str) -> str:
        """Generate one encoded label/description, falling back to keywords."""
        try:
            response = generate_llm_label(
                self.embeddings_manager,
                self.system_prompt,
                prompt,
                max_tokens=DEFAULT_LABEL_MAX_TOKENS,
            )
        except Exception as exc:
            status_code = _status_code(exc)
            logger.warning(
                "LLM label failed for topic %s (model=%s, status=%s): %s; using keyword fallback",
                topic,
                get_config().cluster_label_model,
                status_code if status_code is not None else "n/a",
                exc,
            )
            logger.debug("Prompt for failed topic %s:\n%s", topic, prompt)
            response = ""

        label, description = parse_llm_label_response(response)
        if not label:
            label = " & ".join(keywords[:3])
        return combine_label_description(label, description)
