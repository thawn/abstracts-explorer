"""
ChromaDB connection and collection management.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings

from abstracts_explorer.embeddings.errors import translate_errors

logger = logging.getLogger(__name__)

_CLIENT_CREATION_LOCK = threading.Lock()


class ChromaMixin:
    """ChromaDB client/collection lifecycle and collection statistics."""

    _client: Optional[Any]
    _collection: Optional[Any]
    _openai_client: Any
    _init_lock: Any
    embedding_db: str
    collection_name: str

    @property
    def client(self) -> Any:
        """
        Get the ChromaDB client, connecting automatically on first access.

        Returns
        -------
        chromadb.Client
            Initialized ChromaDB client instance.

        Raises
        ------
        EmbeddingsError
            If connecting to ChromaDB fails.
        """
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    self._connect()
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    @property
    def collection(self) -> Any:
        """
        Get the ChromaDB collection, creating it automatically on first access.

        Calling this property for the first time also connects the client if
        it has not been initialized yet.

        Returns
        -------
        chromadb.Collection
            Initialized ChromaDB collection.

        Raises
        ------
        EmbeddingsError
            If connecting to ChromaDB or creating the collection fails.
        """
        if self._collection is None:
            with self._init_lock:
                if self._collection is None:
                    self.create_collection()
        return self._collection

    @collection.setter
    def collection(self, value: Any) -> None:
        self._collection = value

    def _connect(self) -> None:
        """Connect to ChromaDB via HTTP for URLs or persistent storage for local paths."""
        with translate_errors("Failed to connect to ChromaDB"):
            parsed = urlparse(self.embedding_db)
            with _CLIENT_CREATION_LOCK:
                if parsed.scheme in ("http", "https"):
                    self._client = chromadb.HttpClient(
                        host=parsed.hostname or "localhost",
                        port=parsed.port or 8000,
                        ssl=parsed.scheme == "https",
                        settings=Settings(anonymized_telemetry=False),
                    )
                    logger.debug("Connected to ChromaDB HTTP service at: %s", self.embedding_db)
                else:
                    chroma_path = Path(self.embedding_db)
                    chroma_path.mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    logger.debug("Connected to ChromaDB at: %s", chroma_path)

    def create_collection(self, reset: bool = False) -> None:
        """
        Create or get ChromaDB collection.

        Parameters
        ----------
        reset : bool, optional
            If True, delete existing collection and create new one, by default False

        Raises
        ------
        EmbeddingsError
            If collection creation fails or not connected.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection(reset=True)
        """
        with self._init_lock, translate_errors("Failed to create collection"):
            if reset:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info("Deleted existing collection: %s", self.collection_name)
                except Exception:
                    pass

            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Paper abstracts and metadata"},
            )
            logger.debug("Created/retrieved collection: %s", self.collection_name)

    def close(self) -> None:
        """
        Close the OpenAI client and reset the ChromaDB connection.

        Safe to call multiple times.
        """
        openai_client = getattr(self, "_openai_client", None)
        if openai_client is not None:
            try:
                openai_client.close()
            except Exception as e:
                logger.debug("Failed to close OpenAI client: %s", e)
            self._openai_client = None
        if self._client is not None:
            self._client = None
            self._collection = None
            logger.debug("ChromaDB connection closed")

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns
        -------
        dict
            Statistics including count, name, and metadata.

        Raises
        ------
        EmbeddingsError
            If collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection()
        >>> stats = em.get_collection_stats()
        >>> print(f"Collection has {stats['count']} papers")
        """
        with translate_errors("Failed to get collection stats"):
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata,
            }

    def _get_records(self, ids: list[str], include: list[str]) -> dict[str, Any]:
        """Fetch records for *ids* from the collection."""
        return self.collection.get(ids=ids, include=include)
