"""
Embedding artifact export/import and deletion.
"""

import logging
from typing import Any, Optional

from abstracts_explorer.db.errors import normalize_model_name
from abstracts_explorer.embeddings.batching import IO_BATCH_SIZE, iter_batches
from abstracts_explorer.embeddings.errors import EmbeddingsError, translate_errors
from abstracts_explorer.embeddings.hashing import is_content_hash
from abstracts_explorer.utils.read_only import require_writable

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1


class ArtifactsMixin:
    """Export, import, and delete embedding records in bulk."""

    collection: Any
    model_name: str
    _store_embeddings: Any

    @require_writable()
    def delete_embeddings_by_ids(self, ids: list[str]) -> int:
        """
        Delete embeddings by paper UID.

        Parameters
        ----------
        ids : list of str
            Paper UIDs to remove.

        Returns
        -------
        int
            Number of embeddings submitted for deletion.

        Raises
        ------
        EmbeddingsError
            If the deletion fails.
        """
        if not ids:
            return 0

        with translate_errors("Failed to delete embeddings"):
            deleted = 0
            for batch in iter_batches(ids, IO_BATCH_SIZE):
                self.collection.delete(ids=list(batch))
                deleted += len(batch)
            from abstracts_explorer.cache import invalidate_result_caches

            invalidate_result_caches()
            return deleted

    def export_embeddings(self, ids: list[str]) -> dict[str, Any]:
        """
        Export embeddings and content hashes for the given paper UIDs.

        Parameters
        ----------
        ids : list of str
            Paper UIDs to export.

        Returns
        -------
        dict
            Dictionary with ``format_version``, ``ids``, ``documents``, and
            ``embeddings`` lists.

        Raises
        ------
        EmbeddingsError
            If the export fails.
        """
        with translate_errors("Failed to export embeddings"):
            exported_ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []

            for batch in iter_batches(ids, IO_BATCH_SIZE):
                results = self.collection.get(ids=list(batch), include=["documents", "embeddings"])
                result_ids = results.get("ids", [])
                result_docs = results.get("documents", [])
                result_embeddings = results.get("embeddings", [])

                by_id = {uid: (result_docs[index], result_embeddings[index]) for index, uid in enumerate(result_ids)}
                for uid in batch:
                    if uid not in by_id:
                        continue
                    document, embedding = by_id[uid]
                    exported_ids.append(uid)
                    documents.append(str(document))
                    vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                    embeddings.append(vector)

            return {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "model": getattr(self, "model_name", None),
                "dimensions": len(embeddings[0]) if embeddings else None,
                "ids": exported_ids,
                "documents": documents,
                "embeddings": embeddings,
            }

    @require_writable()
    def import_embeddings(
        self,
        data: dict[str, Any],
        replace_ids: Optional[list[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Import embeddings in the current artifact format.

        Parameters
        ----------
        data : dict
            Dictionary with ``format_version``, ``ids``, ``documents``, and
            ``embeddings`` lists.
        replace_ids : list of str, optional
            Existing embedding IDs to delete before importing.
        batch_size : int, optional
            Number of embeddings to add per batch.

        Returns
        -------
        int
            Number of embeddings imported.

        Raises
        ------
        EmbeddingsError
            If the artifact format is invalid or the import fails.
        """
        if data.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise EmbeddingsError("Unsupported embeddings artifact format")

        artifact_model = data.get("model")
        if artifact_model and getattr(self, "model_name", None):
            if normalize_model_name(str(artifact_model)) != normalize_model_name(self.model_name):
                raise EmbeddingsError(
                    f"Embeddings artifact was created with model '{artifact_model}', "
                    f"but the current model is '{self.model_name}'."
                )

        ids = data.get("ids", [])
        documents = data.get("documents", [])
        embeddings = data.get("embeddings", [])
        if not ids:
            return 0
        if not (len(ids) == len(documents) == len(embeddings)):
            raise EmbeddingsError("Embeddings artifact fields have inconsistent lengths")
        if not all(is_content_hash(document) for document in documents):
            raise EmbeddingsError("Embeddings artifact contains invalid content hashes")

        with translate_errors("Failed to import embeddings"):
            backup = self.export_embeddings(list(replace_ids)) if replace_ids else None
            if replace_ids:
                self.delete_embeddings_by_ids(replace_ids)

            imported = 0
            try:
                for batch in iter_batches(list(zip(ids, documents, embeddings)), batch_size):
                    self._store_embeddings(
                        ids=[item[0] for item in batch],
                        documents=[item[1] for item in batch],
                        embeddings=[item[2] for item in batch],
                    )
                    imported += len(batch)
            except Exception:
                if replace_ids:
                    try:
                        self.delete_embeddings_by_ids([str(uid) for uid in ids])
                        if backup and backup.get("ids"):
                            self.import_embeddings(backup)
                    except Exception as restore_error:
                        logger.error("Failed to restore previous embeddings after import failure: %s", restore_error)
                raise
            return imported
