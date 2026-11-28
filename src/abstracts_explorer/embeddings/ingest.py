"""
Embedding ingestion: embed papers from the database and upsert vectors.
"""

import logging
from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings.batching import EMBED_BATCH_SIZE, EMBEDDING_FETCH_BATCH_SIZE, iter_batches
from abstracts_explorer.embeddings.errors import translate_errors
from abstracts_explorer.embeddings.hashing import hash_text
from abstracts_explorer.utils.disk_utils import ensure_free_space
from abstracts_explorer.utils.http_utils import is_llm_rate_limit_error
from abstracts_explorer.utils.read_only import require_writable

logger = logging.getLogger(__name__)


class IngestMixin:
    """Store embeddings for papers and keep them in sync with content hashes."""

    collection: Any
    model_name: str
    embedding_db: str
    min_free_disk_mb: int
    generate_embedding: Any
    generate_embeddings: Any
    _get_records: Any

    def _ensure_disk_space(self) -> None:
        """Refuse to write embeddings when the local filesystem is nearly full."""
        if self.embedding_db.startswith(("http://", "https://")):
            return
        min_free_mb = getattr(self, "min_free_disk_mb", 0)
        if min_free_mb <= 0:
            return
        ensure_free_space(
            self.embedding_db,
            min_free_mb * 1024 * 1024,
            description="writing embeddings",
        )

    @staticmethod
    def embedding_text_from_paper(paper: dict) -> str:
        """
        Extract text for embedding from a paper dictionary.

        Parameters
        ----------
        paper : dict
            Dictionary containing paper information.

        Returns
        -------
        str
            Text to be used for embedding.
        """
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        embedding_text = f"{title}\n\n{abstract}".strip()
        if not embedding_text:
            raise ValueError(f"Cannot create embedding text for paper {paper['uid']}: no abstract and no title")
        return embedding_text

    @require_writable()
    def _store_embeddings(self, ids: list[str], documents: list[str], embeddings: Sequence[Any]) -> None:
        """Create or replace records in the collection (upsert semantics)."""
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents)
        from abstracts_explorer.cache import invalidate_result_caches

        invalidate_result_caches()

    @require_writable()
    def _embed_and_upsert(self, uid: str, text: str) -> None:
        """Embed *text* and upsert it for *uid* with its content hash."""
        embedding = self.generate_embedding(text)
        self._store_embeddings(ids=[uid], documents=[hash_text(text)], embeddings=[embedding])

    def _group_papers_for_embedding(
        self, rows: list[dict]
    ) -> tuple[dict[str, str], list[str], list[str], dict[str, str]]:
        """
        Group paper rows so each unique abstract is embedded only once.

        Rows sharing a ``content_hash`` within the same conference and year
        are duplicate presentations of the same abstract.  The first row of
        each group is the *representative* whose text is embedded; the
        remaining rows are *duplicates* that reuse the representative vector.

        Parameters
        ----------
        rows : list of dict
            Paper rows from the database.

        Returns
        -------
        tuple
            ``(texts, representatives, duplicates, representative_of)`` where
            *texts* maps every embeddable UID to its embedding text,
            *representatives* is the ordered list of UIDs to embed,
            *duplicates* is the ordered list of UIDs that reuse a vector, and
            *representative_of* maps each duplicate UID to its representative.
        """
        texts: dict[str, str] = {}
        representatives: list[str] = []
        duplicates: list[str] = []
        representative_of: dict[str, str] = {}
        representative_by_key: dict[tuple, str] = {}
        seen: set[str] = set()

        for paper in rows:
            uid = paper["uid"]
            if uid in seen:
                continue
            seen.add(uid)
            try:
                text = self.embedding_text_from_paper(paper)
            except ValueError as e:
                logger.error("Failed to embed paper %s: %s", uid, e)
                continue
            texts[uid] = text

            key = (paper.get("conference"), paper.get("year"), paper.get("content_hash") or uid)
            representative = representative_by_key.get(key)
            if representative is None:
                representative_by_key[key] = uid
                representatives.append(uid)
            else:
                duplicates.append(uid)
                representative_of[uid] = representative

        return texts, representatives, duplicates, representative_of

    def _embed_uids(self, uids: list[str], texts: dict[str, str]) -> int:
        """
        Embed *uids*, sending each unique text once per batch.

        The resulting vector is stored for every UID that shares the text so
        that duplicate presentations still have their own record.

        Parameters
        ----------
        uids : list of str
            UIDs to embed.
        texts : dict
            Mapping of UID to embedding text.

        Returns
        -------
        int
            Number of UIDs successfully embedded.
        """
        embedded = 0
        for batch in iter_batches(uids, EMBED_BATCH_SIZE):
            groups: dict[str, list[str]] = {}
            for uid in batch:
                groups.setdefault(texts[uid], []).append(uid)
            unique_texts = list(groups)
            try:
                vectors = self.generate_embeddings(unique_texts)
                vector_by_text = dict(zip(unique_texts, vectors))
                batch_ids = list(batch)
                self._store_embeddings(
                    ids=batch_ids,
                    documents=[hash_text(texts[uid]) for uid in batch_ids],
                    embeddings=[vector_by_text[texts[uid]] for uid in batch_ids],
                )
                embedded += len(batch_ids)
            except Exception as batch_error:
                if is_llm_rate_limit_error(batch_error):
                    logger.error(
                        "Embedding batch was rate-limited (%s); aborting instead of " "retrying papers individually",
                        batch_error,
                    )
                    raise
                logger.error("Failed to embed batch (%s); retrying papers individually", batch_error)
                unique_vectors: dict[str, list[float]] = {}
                for uid in batch:
                    text = texts[uid]
                    try:
                        vector = unique_vectors.get(text)
                        if vector is None:
                            vector = self.generate_embedding(text)
                            unique_vectors[text] = vector
                        self._store_embeddings(ids=[uid], documents=[hash_text(text)], embeddings=[vector])
                        embedded += 1
                    except Exception as e:
                        logger.error("Failed to embed paper %s: %s", uid, e)
        return embedded

    def _copy_representative_embeddings(self, duplicate_groups: dict[str, list[str]], texts: dict[str, str]) -> int:
        """
        Store duplicate UIDs using their representative's existing vector.

        Parameters
        ----------
        duplicate_groups : dict
            Mapping of representative UID to the duplicate UIDs that reuse it.
        texts : dict
            Mapping of UID to embedding text (used for the stored content hash).

        Returns
        -------
        int
            Number of duplicate UIDs successfully stored.
        """
        if not duplicate_groups:
            return 0

        vectors_by_representative: dict[str, Any] = {}
        for batch in iter_batches(list(duplicate_groups), EMBEDDING_FETCH_BATCH_SIZE):
            try:
                records = self._get_records(list(batch), ["embeddings"])
            except Exception as e:
                logger.warning("Error fetching representative embeddings: %s", e)
                continue
            ids = records.get("ids") or []
            embeddings = records.get("embeddings")
            if embeddings is None:
                continue
            for representative_uid, vector in zip(ids, embeddings):
                vectors_by_representative[str(representative_uid)] = np.asarray(vector, dtype=np.float32)

        copied = 0
        pending_ids: list[str] = []
        pending_documents: list[str] = []
        pending_vectors: list[list[float]] = []

        def flush() -> None:
            nonlocal copied, pending_ids, pending_documents, pending_vectors
            if not pending_ids:
                return
            self._store_embeddings(
                ids=pending_ids,
                documents=pending_documents,
                embeddings=pending_vectors,
            )
            copied += len(pending_ids)
            pending_ids, pending_documents, pending_vectors = [], [], []

        for representative, duplicate_uids in duplicate_groups.items():
            vector = vectors_by_representative.get(representative)
            if vector is None:
                logger.warning(
                    "Skipping %d duplicate embedding(s): representative %s has no stored embedding",
                    len(duplicate_uids),
                    representative,
                )
                continue
            for uid in duplicate_uids:
                pending_ids.append(uid)
                pending_documents.append(hash_text(texts[uid]))
                pending_vectors.append(vector)
                if len(pending_ids) >= EMBED_BATCH_SIZE:
                    flush()
        flush()
        return copied

    @require_writable()
    def add_paper(self, paper: dict) -> None:
        """
        Add a paper to the vector database.

        If an embedding for the paper's UID already exists, it is replaced.

        Parameters
        ----------
        paper : dict
            Dictionary containing paper information. Must follow the paper database schema.

        Raises
        ------
        EmbeddingsError
            If adding paper fails or collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection()
        >>> em.add_paper(paper_dict)
        """
        with translate_errors(f"Failed to add paper {paper['uid']}"):
            embedding_text = self.embedding_text_from_paper(paper)
            self._embed_and_upsert(paper["uid"], embedding_text)
            logger.debug("Added paper %s to collection", paper["uid"])

    @require_writable()
    def embed_from_database(
        self,
        where_filters: Optional[dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_recreate: bool = False,
    ) -> int:
        """
        Embed papers from the database.

        Reads papers from the database and generates embeddings for their abstracts.

        Parameters
        ----------
        where_filters : dict, optional
            Structured filters passed to
            :meth:`~abstracts_explorer.database.DatabaseManager.get_paper`
            (e.g. ``{"conference": "NeurIPS", "year": 2025}``).
        progress_callback : callable, optional
            Callback function to report progress. Called with (current, total) number of papers processed.
        force_recreate : bool, optional
            If True, skip checking for existing embeddings and recreate all, by default False

        Returns
        -------
        int
            Number of papers successfully embedded.

        Raises
        ------
        EmbeddingsError
            If database reading or embedding fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection()
        >>> count = em.embed_from_database()
        >>> print(f"Embedded {count} papers")
        >>> # Only embed accepted papers
        >>> count = em.embed_from_database(where_filters={"award": {"$ne": None}})
        """
        with translate_errors("Failed to embed from database"):
            self._ensure_disk_space()
            with DatabaseManager() as db_manager:
                db_manager.set_embedding_model(self.model_name)
                rows = db_manager.get_paper(where_filters, limit=0)
                total = len(rows)
                logger.debug("Found %d papers to embed", total)

                texts, representatives, duplicates, representative_of = self._group_papers_for_embedding(rows)

                embedded_count = 0
                skipped_count = 0
                processed = 0

                for batch in iter_batches(representatives, EMBED_BATCH_SIZE):
                    update_uids = (
                        list(batch)
                        if force_recreate
                        else self._texts_needing_update({uid: texts[uid] for uid in batch})
                    )
                    skipped_count += len(batch) - len(update_uids)
                    if update_uids:
                        embedded_count += self._embed_uids(update_uids, texts)
                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total)

                for batch in iter_batches(duplicates, EMBED_BATCH_SIZE):
                    update_uids = (
                        list(batch)
                        if force_recreate
                        else self._texts_needing_update({uid: texts[uid] for uid in batch})
                    )
                    skipped_count += len(batch) - len(update_uids)
                    if update_uids:
                        duplicate_groups: dict[str, list[str]] = {}
                        for uid in update_uids:
                            duplicate_groups.setdefault(representative_of[uid], []).append(uid)
                        embedded_count += self._copy_representative_embeddings(duplicate_groups, texts)
                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total)

                if progress_callback and processed < total:
                    progress_callback(total, total)

            logger.info("Successfully embedded %d papers, skipped %d existing papers", embedded_count, skipped_count)
            return embedded_count

    def _texts_needing_update(self, texts: dict[str, str]) -> list[str]:
        """Return UIDs whose stored content hash differs from *texts*."""
        uids = list(texts)
        try:
            existing = self._get_records(uids, ["documents"])
        except Exception as e:
            logger.warning("Error checking existing embeddings: %s", e)
            return uids

        stored = dict(zip(existing.get("ids", []), existing.get("documents", [])))
        try:
            with DatabaseManager() as db_manager:
                content_hashes = db_manager.get_content_hashes(uids)
        except Exception as e:
            logger.warning("Error reading stored content hashes: %s", e)
            content_hashes = {}

        needing: list[str] = []
        for uid in uids:
            expected = content_hashes.get(uid) or hash_text(texts[uid])
            if stored.get(uid) != expected:
                needing.append(uid)
        return needing
