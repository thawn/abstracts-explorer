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
        self.flat_topics: Optional[Sequence[int]] = None
        self._deferred: Optional[List[Tuple[frozenset[int], List[str], str]]] = None

    def set_flat_topics(self, topics: Sequence[int]) -> None:
        """
        Provide the flat topic assignment per document.

        The mapping is needed to recover the member flat topics of a merged
        hierarchy node from a document selection.

        Parameters
        ----------
        topics : sequence of int
            Flat topic ID for each document, indexed by document position.
        """
        self.flat_topics = list(topics)

    def begin_deferred(self) -> None:
        """Defer LLM labeling until :meth:`flush_deferred` is called.

        While deferred, :meth:`extract_topics` records the clusters that need
        labels and returns cheap keyword placeholders immediately. This lets
        BERTopic build the hierarchy quickly and allows the recorded clusters to
        be labeled in a single parallel batch afterwards.
        """
        self._deferred = []

    def flush_deferred(self, show_progress: Optional[bool] = None) -> Dict[frozenset[int], str]:
        """
        Label all clusters recorded since :meth:`begin_deferred` in one batch.

        Parameters
        ----------
        show_progress : bool or None, optional
            Override progress-bar visibility for the batch. ``None`` (default)
            uses the instance ``show_progress`` setting.

        Returns
        -------
        dict
            Mapping of member flat-topic sets to encoded label/description
            strings.
        """
        recorded = self._deferred or []
        self._deferred = None
        if not recorded:
            return {}

        show = self.show_progress if show_progress is None else show_progress
        pending = [(index, keywords, prompt) for index, (_, keywords, prompt) in enumerate(recorded)]
        progress = (
            tqdm(total=len(pending), desc="Labeling hierarchy clusters (LLM)", unit="cluster")
            if show and len(pending) > 1
            else None
        )
        try:
            labeled = self._label_pending(pending, progress)
        finally:
            if progress is not None:
                progress.close()

        resolved: Dict[frozenset[int], str] = {}
        for index, (members, _, _) in enumerate(recorded):
            if members:
                resolved[members] = labeled[index]
        return resolved

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
        deferring = self._deferred is not None
        use_progress = not deferring and self.show_progress and len(repr_docs_mappings) > 1
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
                prompt = self._build_prompt(keywords, docs)

                if self._deferred is not None:
                    self._deferred.append((self._members_for_topic(documents, topic), keywords, prompt))
                    placeholder = " & ".join(keywords[:3]) or "Merged cluster"
                    updated_topics[topic] = [(placeholder, 1)]
                    continue

                pending.append((topic, keywords, prompt))

            if pending:
                labeled = self._label_pending(pending, progress)
                for topic, encoded in labeled.items():
                    updated_topics[topic] = [(encoded, 1)]
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

    def _label_pending(
        self,
        pending: Sequence[Tuple[Any, Sequence[str], str]],
        progress: Any = None,
    ) -> Dict[Any, str]:
        """
        Label a batch of pending clusters, in parallel when configured.

        Parameters
        ----------
        pending : sequence of tuple
            ``(key, keywords, prompt)`` items to label. The key is only used to
            map results back and may be an arbitrary hashable value.
        progress : tqdm or None, optional
            Progress bar to update once per completed item.

        Returns
        -------
        dict
            Mapping of each key to its encoded label/description string.
        """
        labeled: Dict[Any, str] = {}
        if not pending:
            return labeled

        max_workers = self._effective_max_workers(len(pending))
        if max_workers <= 1:
            for key, keywords, prompt in pending:
                labeled[key] = self._label_topic(key, keywords, prompt)
                if progress is not None:
                    progress.update(1)
            return labeled

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cluster-label",
        ) as executor:
            futures = {
                executor.submit(self._label_topic, key, keywords, prompt): key for key, keywords, prompt in pending
            }
            for future in as_completed(futures):
                key = futures[future]
                labeled[key] = future.result()
                if progress is not None:
                    progress.update(1)
        return labeled

    def _members_for_topic(self, documents: Any, topic: Any) -> frozenset[int]:
        """
        Return the flat topics covered by a hierarchy merge selection.

        BERTopic relabels every document in a merged selection as topic ``0``
        but keeps the original ``ID`` column, so the flat topic can be recovered
        through :attr:`flat_topics`.

        Parameters
        ----------
        documents : pandas.DataFrame
            Selection of documents for the merged cluster.
        topic : Any
            Topic ID within *documents* (``0`` for hierarchy merges).

        Returns
        -------
        frozenset of int
            Member flat topic IDs, excluding outliers. Empty when the selection
            or flat-topic mapping is unavailable.
        """
        if not self.flat_topics:
            return frozenset()
        try:
            selection = documents.loc[documents["Topic"] == topic, :]
            if "ID" in selection.columns:
                document_ids = [int(index) for index in selection["ID"].tolist()]
            else:
                document_ids = [int(index) for index in selection.index.tolist()]
        except Exception:
            return frozenset()
        return frozenset(
            int(self.flat_topics[document_id])
            for document_id in document_ids
            if 0 <= document_id < len(self.flat_topics) and int(self.flat_topics[document_id]) >= 0
        )

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
