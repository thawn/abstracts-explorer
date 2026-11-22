"""
Project LLM representation model for BERTopic.
"""

import logging
from typing import Any, Dict, List, Mapping, Tuple

from bertopic.representation import BaseRepresentation
from tqdm import tqdm

from abstracts_explorer.clustering.labels import combine_label_description, parse_llm_label_response
from abstracts_explorer.clustering.llm import DEFAULT_LABEL_MAX_TOKENS, generate_llm_label

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
        """
        self.embeddings_manager = embeddings_manager
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.show_progress = show_progress

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
                docs_text = "\n".join(f"- {doc[: self.doc_length]}" for doc in docs)
                prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
                prompt = prompt.replace("[DOCUMENTS]", docs_text)
                try:
                    response = generate_llm_label(
                        self.embeddings_manager,
                        self.system_prompt,
                        prompt,
                        max_tokens=DEFAULT_LABEL_MAX_TOKENS,
                    )
                except Exception as exc:
                    logger.warning(
                        "LLM label failed for topic %s: %s; using keyword fallback. Prompt:\n%s",
                        topic,
                        exc,
                        prompt,
                    )
                    response = ""

                label, description = parse_llm_label_response(response)
                if not label:
                    label = " & ".join(keywords[:3])
                updated_topics[topic] = [(combine_label_description(label, description), 1)]
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return updated_topics
