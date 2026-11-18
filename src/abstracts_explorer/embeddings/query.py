"""
Similarity queries against the ChromaDB collection.
"""

import logging
import threading
from itertools import takewhile
from typing import Any, Optional

from abstracts_explorer.embeddings.batching import ID_BATCH_SIZE, MAX_QUERY_RESULTS, iter_batches
from abstracts_explorer.embeddings.errors import EmbeddingsError, translate_errors

logger = logging.getLogger(__name__)

PaperMatch = tuple[str, float]


class QueryEmbedding:
    """
    Lazily generated query embedding bound to an embeddings manager.

    The embedding vector is generated on first access and cached, so the
    same object can be reused across distance filtering, counting via
    ``len(matches)``, and result hydration without duplicate embedding API
    calls.

    Parameters
    ----------
    text : str
        Query text to embed.
    embeddings_manager : EmbeddingsManager
        Manager used to generate the embedding vector.

    Examples
    --------
    >>> query = QueryEmbedding("graph neural networks", em)
    >>> query.embedding  # generated once
    >>> em.search_similar(query, limit=5)
    """

    __slots__ = ("text", "_manager", "_vector", "_lock")

    def __init__(self, text: str, embeddings_manager: Any) -> None:
        self.text = text
        self._manager = embeddings_manager
        self._vector: Optional[list[float]] = None
        self._lock = threading.Lock()

    @property
    def embeddings_manager(self) -> Any:
        """Return the embeddings manager bound to this query."""
        return self._manager

    @property
    def embedding(self) -> list[float]:
        """
        Return the embedding vector, generating it on first access.

        Returns
        -------
        list of float
            Query embedding vector.
        """
        if self._vector is None:
            with self._lock:
                if self._vector is None:
                    self._vector = self._manager.generate_embedding(self.text)
        assert self._vector is not None
        return self._vector


class QueryMixin:
    """Run embedding queries and return flat results."""

    collection: Any
    generate_embedding: Any

    def _query_embeddings(
        self,
        query: QueryEmbedding,
        *,
        ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[PaperMatch]:
        """
        Run a distance query against the collection.

        The number of results is capped by ``limit`` (when given), the
        number of candidate IDs, the collection size, and
        :data:`~abstracts_explorer.embeddings.batching.MAX_QUERY_RESULTS`.
        When no candidates exist, an empty list is returned without querying.

        Parameters
        ----------
        query : QueryEmbedding
            Query embedding object. Its vector is generated lazily once.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        limit : int, optional
            Maximum number of results to return.

        Returns
        -------
        list of tuple
            Ordered ``(paper_uid, distance)`` pairs.

        Raises
        ------
        EmbeddingsError
            If the query text is empty or the query itself fails.
        """
        if not query.text or not query.text.strip():
            raise EmbeddingsError("Query cannot be empty")

        query_embedding = query.embedding
        candidate_count = len(ids) if ids is not None else self.collection.count()
        request_limit = min(
            limit if limit is not None else candidate_count,
            candidate_count,
            MAX_QUERY_RESULTS,
        )
        if request_limit <= 0:
            return []

        if ids is None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=request_limit,
                include=["distances"],
            )
            paper_ids = list(results["ids"][0]) if results.get("ids") else []
            distances = list(results["distances"][0]) if results.get("distances") else []
            return list(zip(paper_ids, distances))

        merged: list[PaperMatch] = []
        for batch in iter_batches(ids, ID_BATCH_SIZE):
            batch_limit = min(request_limit, len(batch))
            if batch_limit <= 0:
                break
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=batch_limit,
                include=["distances"],
                ids=list(batch),
            )
            batch_ids = list(results["ids"][0]) if results.get("ids") else []
            batch_distances = list(results["distances"][0]) if results.get("distances") else []
            merged.extend(zip(batch_ids, batch_distances))

        merged.sort(key=lambda item: item[1])
        return merged[:request_limit]

    def search_similar(
        self,
        query: QueryEmbedding,
        limit: int = 10,
        ids: Optional[list[str]] = None,
    ) -> list[PaperMatch]:
        """
        Search for similar papers using semantic similarity.

        Parameters
        ----------
        query : QueryEmbedding
            Query embedding object.
        limit : int, optional
            Number of results to return, by default 10.
        ids : list of str, optional
            Restrict the search to these paper UIDs. An empty list returns an
            empty result.

        Returns
        -------
        list of tuple
            Ordered ``(paper_uid, distance)`` pairs.

        Raises
        ------
        EmbeddingsError
            If search fails or collection not initialized.

        Examples
        --------
        >>> query = QueryEmbedding("deep learning transformers", em)
        >>> results = em.search_similar(query, limit=5)
        >>> for paper_id, distance in results:
        ...     print(f"{paper_id}: {distance}")
        """
        if not query.text or not query.text.strip():
            raise EmbeddingsError("Query cannot be empty")
        if ids is not None and not ids:
            return []

        with translate_errors("Failed to search"):
            result = self._query_embeddings(query, ids=ids, limit=limit)
            logger.info("Found %d similar papers", len(result))
            return result

    def query_ids_within_distance(
        self,
        query: QueryEmbedding,
        distance_threshold: float = 1.1,
        ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[PaperMatch]:
        """
        Return paper matches within a distance threshold.

        Parameters
        ----------
        query : QueryEmbedding
            Query embedding object.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        limit : int, optional
            Maximum number of matches to return.  When omitted, all matches
            within the threshold are returned.

        Returns
        -------
        list of tuple
            Ordered ``(paper_uid, distance)`` pairs within the threshold.

        Raises
        ------
        EmbeddingsError
            If the query fails.
        """
        if ids is not None and not ids:
            return []
        if self.collection.count() == 0:
            return []

        # ChromaDB has no distance-threshold parameter: it returns the nearest
        # neighbors first. Request 2x the requested match limit so distance
        # filtering still has enough candidates to return up to `limit` matches.
        request_limit = None if limit is None else limit * 2

        with translate_errors("Failed to query papers within distance"):
            result = self._query_embeddings(query, ids=ids, limit=request_limit)
            matches = list(takewhile(lambda pair: pair[1] <= distance_threshold, result))
            return matches[:limit] if limit is not None else matches
