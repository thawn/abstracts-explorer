"""
Project LLM representation model for BERTopic.
"""

import logging
from typing import Any, Dict, List, Mapping, Tuple

from bertopic.representation import BaseRepresentation
from tqdm import tqdm

from abstracts_explorer.clustering.llm import generate_llm_label

logger = logging.getLogger(__name__)

LLM_LABEL_PROMPT = """You are labeling a cluster of research papers.

Sample papers:
[DOCUMENTS]

Top keywords: [KEYWORDS]

Generate a concise, descriptive topic label of 3-6 words. Use title case.
Return only the label, without quotes, a "topic:" prefix, or explanation."""

LLM_LABEL_SYSTEM_PROMPT = (
    "You are a research paper categorization expert. Generate concise, descriptive labels for clusters of papers."
)


class ProjectLLMRepresentation(BaseRepresentation):
    """Generate one short topic label per cluster with the project LLM."""

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

    def extract_topics(
        self,
        topic_model: Any,
        documents: Any,
        c_tf_idf: Any,
        topics: Mapping[Any, List[Tuple[str, float]]],
    ) -> Mapping[Any, List[Tuple[str, float]]]:
        """
        Generate LLM labels for the supplied topics.

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
            Updated topic representations mapping topic ID to one label tuple.
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
                keywords = [word for word, _ in topics[topic][:10]]
                docs_text = "\n".join(f"- {doc[: self.doc_length]}" for doc in docs)
                prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
                prompt = prompt.replace("[DOCUMENTS]", docs_text)
                try:
                    label = generate_llm_label(self.embeddings_manager, self.system_prompt, prompt, max_tokens=200)
                except Exception as exc:
                    logger.warning("LLM label failed for topic %s: %s; using keyword fallback", topic, exc)
                    label = ""

                label = " ".join(label.split()).strip("\"'")
                if label.lower().startswith("topic:"):
                    label = label.split(":", 1)[1].strip()
                if not label:
                    label = " & ".join(keywords[:3])
                updated_topics[topic] = [(label[:120], 1)]
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return updated_topics
