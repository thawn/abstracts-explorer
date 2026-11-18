"""
Embedding ingestion: embed papers from the database and upsert vectors.
"""

import logging
from typing import Any, Callable, Optional

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings.batching import EMBED_BATCH_SIZE, iter_batches
from abstracts_explorer.embeddings.errors import translate_errors
from abstracts_explorer.embeddings.hashing import hash_text

logger = logging.getLogger(__name__)


class IngestMixin:
    """Store embeddings for papers and keep them in sync with content hashes."""

    collection: Any
    model_name: str
    generate_embedding: Any
    generate_embeddings: Any
    _get_records: Any

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

    def _store_embeddings(self, ids: list[str], documents: list[str], embeddings: list[list[float]]) -> None:
        """Create or replace records in the collection (upsert semantics)."""
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents)

    def _embed_and_upsert(self, uid: str, text: str) -> None:
        """Embed *text* and upsert it for *uid* with its content hash."""
        embedding = self.generate_embedding(text)
        self._store_embeddings(ids=[uid], documents=[hash_text(text)], embeddings=[embedding])

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
            with DatabaseManager() as db_manager:
                db_manager.set_embedding_model(self.model_name)
                rows = db_manager.get_paper(where_filters, limit=0)
                total = len(rows)
                logger.debug("Found %d papers to embed", total)

                embedded_count = 0
                skipped_count = 0
                processed = 0
                for batch in iter_batches(rows, EMBED_BATCH_SIZE):
                    papers: dict[str, dict] = {}
                    texts: dict[str, str] = {}
                    for paper in batch:
                        try:
                            texts[paper["uid"]] = self.embedding_text_from_paper(paper)
                            papers[paper["uid"]] = paper
                        except ValueError as e:
                            logger.error("Failed to embed paper %s: %s", paper.get("uid"), e)

                    update_uids = list(texts) if force_recreate else self._texts_needing_update(texts)
                    skipped_count += len(texts) - len(update_uids)
                    if update_uids:
                        try:
                            vectors = self.generate_embeddings([texts[uid] for uid in update_uids])
                            self._store_embeddings(
                                ids=update_uids,
                                documents=[hash_text(texts[uid]) for uid in update_uids],
                                embeddings=vectors,
                            )
                            embedded_count += len(update_uids)
                        except Exception as batch_error:
                            logger.error("Failed to embed batch (%s); retrying papers individually", batch_error)
                            for uid in update_uids:
                                try:
                                    self._embed_and_upsert(uid, texts[uid])
                                    embedded_count += 1
                                except Exception as e:
                                    logger.error("Failed to embed paper %s: %s", uid, e)

                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total)

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
        return [uid for uid in uids if stored.get(uid) != hash_text(texts[uid])]
