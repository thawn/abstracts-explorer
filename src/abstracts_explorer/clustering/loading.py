"""
Loading embeddings and paper metadata from ChromaDB and the database.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.embeddings.batching import EMBEDDING_FETCH_BATCH_SIZE, iter_batches
from abstracts_explorer.profiling import profile_span

logger = logging.getLogger(__name__)


class LoadingMixin:
    """Load embeddings and their paper metadata."""

    embeddings_manager: Any
    database: Any
    embeddings: Any
    paper_ids: Any
    metadatas: Any

    def load_embeddings(
        self,
        limit: Optional[int] = None,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> int:
        """
        Load embeddings from ChromaDB collection.

        Parameters
        ----------
        limit : int, optional
            Maximum number of embeddings to load. If None, load all.
        conferences : list of str, optional
            Filter to only load embeddings for these conferences.
        years : list of int, optional
            Filter to only load embeddings for these years.

        Returns
        -------
        int
            Number of embeddings loaded

        Raises
        ------
        ClusteringError
            If loading fails or collection is empty
        """
        if not self.embeddings_manager.collection:
            raise ClusteringError("Collection not initialized in embeddings manager")

        try:
            candidate_ids = self._resolve_candidate_ids(conferences, years)
            chroma_ids, chroma_embeddings = self._fetch_chroma_embeddings(candidate_ids, limit)
            summaries, keep_indices = self._load_summaries(chroma_ids)
            paper_ids, embeddings, metadatas = self._dedupe_presentations(
                chroma_ids,
                chroma_embeddings,
                summaries,
                keep_indices,
            )

            self.paper_ids = paper_ids
            self.embeddings = np.asarray(embeddings, dtype=np.float32)
            self.metadatas = metadatas

            logger.info(f"Loaded {len(self.paper_ids)} embeddings with dimension {self.embeddings.shape[1]}")
            return len(self.paper_ids)

        except ClusteringError:
            raise
        except Exception as e:
            raise ClusteringError(f"Failed to load embeddings: {str(e)}") from e

    def _resolve_candidate_ids(
        self,
        conferences: Optional[List[str]],
        years: Optional[List[int]],
    ) -> Optional[List[str]]:
        """Return candidate paper UIDs, or None when no filters are applied."""
        if not (conferences or years):
            return None
        ids = self.database.resolve_paper_uids(conferences=conferences, years=years)
        if not ids:
            raise ClusteringError("No embeddings found in collection")
        return ids

    def _fetch_all_embeddings(self, limit: Optional[int]) -> tuple[List[str], Any]:
        """Fetch embeddings for the whole collection, optionally limited."""
        get_kwargs: Dict[str, Any] = {"include": ["embeddings"]}
        if limit:
            get_kwargs["limit"] = limit
        with profile_span("chroma.get", "chroma", ids=None, limit=limit):
            results = self.embeddings_manager.collection.get(**get_kwargs)
        embeddings = results.get("embeddings")
        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float32)
        return results.get("ids") or [], embeddings

    def _fetch_embeddings_by_ids(self, ids: List[str], limit: Optional[int]) -> tuple[List[str], List[Any]]:
        """Fetch embeddings for explicit paper UIDs in batches."""
        batched_ids: List[str] = []
        batched_embeddings: List[Any] = []
        remaining = limit
        for batch in iter_batches(ids, EMBEDDING_FETCH_BATCH_SIZE):
            batch_kwargs: Dict[str, Any] = {"include": ["embeddings"], "ids": list(batch)}
            if remaining:
                batch_kwargs["limit"] = remaining
            with profile_span("chroma.get", "chroma", ids=len(batch), limit=remaining):
                batch_results = self.embeddings_manager.collection.get(**batch_kwargs)
            batch_ids = batch_results.get("ids") or []
            batch_embeddings = batch_results.get("embeddings")
            if batch_embeddings is None:
                continue
            batched_ids.extend(batch_ids)
            batched_embeddings.extend(np.asarray(batch_embeddings, dtype=np.float32))
            if remaining:
                remaining -= len(batch_ids)
                if remaining <= 0:
                    break
        return batched_ids, batched_embeddings

    def _fetch_chroma_embeddings(
        self,
        ids: Optional[List[str]],
        limit: Optional[int],
    ) -> tuple[List[str], Any]:
        """Fetch and validate Chroma embeddings for all or selected papers."""
        if ids is None:
            chroma_ids, chroma_embeddings = self._fetch_all_embeddings(limit)
        else:
            chroma_ids, chroma_embeddings = self._fetch_embeddings_by_ids(ids, limit)
        if not chroma_ids or chroma_embeddings is None:
            raise ClusteringError("No embeddings found in collection")
        return chroma_ids, chroma_embeddings

    def _load_summaries(self, chroma_ids: List[str]) -> tuple[Dict[str, Dict[str, Any]], List[int]]:
        """Load paper summaries and return the indices with matching rows."""
        summaries = {paper["uid"]: paper for paper in self.database.get_paper_summaries_by_uids(list(chroma_ids))}
        keep_indices = [index for index, uid in enumerate(chroma_ids) if uid in summaries]
        if not keep_indices:
            raise ClusteringError("No matching papers found in the database")
        if len(keep_indices) != len(chroma_ids):
            logger.warning(
                "%d embedding(s) have no matching paper row and will be ignored",
                len(chroma_ids) - len(keep_indices),
            )
        return summaries, keep_indices

    def _dedupe_presentations(
        self,
        chroma_ids: List[str],
        chroma_embeddings: Any,
        summaries: Dict[str, Dict[str, Any]],
        keep_indices: List[int],
    ) -> tuple[List[str], List[Any], List[Dict[str, Any]]]:
        """Drop duplicate presentations and attach location metadata."""
        kept_ids = [chroma_ids[index] for index in keep_indices]
        content_hashes = [str(summaries[uid].get("content_hash") or uid) for uid in kept_ids]
        locations_by_hash = self.database.get_locations_by_content_hashes(content_hashes)

        seen_hashes: set[tuple] = set()
        deduped_ids: List[str] = []
        deduped_embeddings: List[Any] = []
        deduped_metadatas: List[Dict[str, Any]] = []
        for index, uid, content_hash in zip(keep_indices, kept_ids, content_hashes):
            metadata = summaries[uid]
            dedupe_key = (metadata.get("conference"), metadata.get("year"), content_hash)
            if dedupe_key in seen_hashes:
                continue
            seen_hashes.add(dedupe_key)

            metadata = dict(metadata)
            locations = locations_by_hash.get(content_hash)
            if not locations:
                single = metadata.get("location")
                locations = [single] if single else []
            metadata["locations"] = locations
            metadata["primary_location"] = locations[0] if locations else metadata.get("location")

            deduped_ids.append(uid)
            deduped_embeddings.append(chroma_embeddings[index])
            deduped_metadatas.append(metadata)

        if len(deduped_ids) != len(kept_ids):
            logger.info(
                "Deduplicated %d presentation row(s) by content hash before clustering",
                len(kept_ids) - len(deduped_ids),
            )
        return deduped_ids, deduped_embeddings, deduped_metadatas
