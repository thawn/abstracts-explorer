"""
Similarity queries against the ChromaDB collection.
"""

import logging
from dataclasses import dataclass
from itertools import takewhile
from typing import Any, Optional

from abstracts_explorer.embeddings.batching import ID_BATCH_SIZE, MAX_QUERY_RESULTS, iter_batches
from abstracts_explorer.embeddings.errors import EmbeddingsError, translate_errors

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Flat result of a distance query against the embedding collection."""

    ids: list[str]
    distances: list[float]
    query_embedding: list[float]


class QueryMixin:
    """Run embedding queries and return flat results."""

    collection: Any
    generate_embedding: Any

    def _query_embeddings(
        self,
        query: str,
        *,
        ids: Optional[list[str]] = None,
        query_embedding: Optional[list[float]] = None,
        n_results: Optional[int] = None,
    ) -> QueryResult:
        """
        Run a distance query against the collection and return flat results.

        Generates the query embedding when not supplied. The number of results
        is capped by ``n_results`` (when given), the number of candidate IDs,
        the collection size, and :data:`~abstracts_explorer.embeddings.batching.MAX_QUERY_RESULTS`.
        When no candidates exist, empty result lists are returned without querying.

        Parameters
        ----------
        query : str
            The search query text.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        query_embedding : list of float, optional
            Pre-computed embedding for the query.
        n_results : int, optional
            Maximum number of results to return.

        Returns
        -------
        QueryResult
            Paper IDs, distances, and the query embedding.

        Raises
        ------
        EmbeddingsError
            If the query is empty or the query itself fails.
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")

        if query_embedding is None:
            query_embedding = self.generate_embedding(query)
        assert query_embedding is not None

        candidate_count = len(ids) if ids is not None else self.collection.count()
        limit = min(
            n_results if n_results is not None else candidate_count,
            candidate_count,
            MAX_QUERY_RESULTS,
        )
        if limit <= 0:
            return QueryResult([], [], query_embedding)

        if ids is None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["distances"],
            )
            paper_ids = list(results["ids"][0]) if results.get("ids") else []
            distances = list(results["distances"][0]) if results.get("distances") else []
            return QueryResult(paper_ids, distances, query_embedding)

        merged: list[tuple[str, float]] = []
        for batch in iter_batches(ids, ID_BATCH_SIZE):
            batch_limit = min(limit, len(batch))
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
        merged = merged[:limit]
        return QueryResult([uid for uid, _ in merged], [distance for _, distance in merged], query_embedding)

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Search for similar papers using semantic similarity.

        Parameters
        ----------
        query : str
            Query text to search for.
        n_results : int, optional
            Number of results to return, by default 10.
        ids : list of str, optional
            Restrict the search to these paper UIDs. An empty list returns an
            empty result.

        Returns
        -------
        dict
            ChromaDB-compatible query results containing ``ids`` and ``distances``.

        Raises
        ------
        EmbeddingsError
            If search fails or collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection()
        >>> results = em.search_similar("deep learning transformers", n_results=5)
        >>> for i, paper_id in enumerate(results['ids'][0]):
        ...     print(f"{i+1}. Paper {paper_id}: {results['distances'][0][i]}")
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")
        if ids is not None and not ids:
            return {"ids": [[]], "distances": [[]]}

        with translate_errors("Failed to search"):
            result = self._query_embeddings(query, ids=ids, n_results=n_results)
            logger.info("Found %d similar papers", len(result.ids))
            return {"ids": [result.ids], "distances": [result.distances]}

    def query_ids_within_distance(
        self,
        query: str,
        distance_threshold: float = 1.1,
        ids: Optional[list[str]] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Return identifiers and distances within a distance threshold.

        Parameters
        ----------
        query : str
            The search query text.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        query_embedding : list of float, optional
            Pre-computed embedding for the query.

        Returns
        -------
        dict
            Dictionary containing ``query_embedding``, ``ids``, ``distances``,
            and ``total_considered``.

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails.
        """
        if ids is not None and not ids:
            raise EmbeddingsError("No papers matched the filters")
        if self.collection.count() == 0:
            raise EmbeddingsError("No papers in collection")

        with translate_errors("Failed to query papers within distance"):
            result = self._query_embeddings(query, ids=ids, query_embedding=query_embedding)
            if not result.ids:
                raise EmbeddingsError("No results from collection query")

            matches = list(takewhile(lambda pair: pair[1] <= distance_threshold, zip(result.ids, result.distances)))
            return {
                "query_embedding": result.query_embedding,
                "ids": [paper_id for paper_id, _ in matches],
                "distances": [float(distance) for _, distance in matches],
                "total_considered": len(result.ids),
            }
