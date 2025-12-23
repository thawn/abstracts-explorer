"""
Embeddings Module
=================

This module provides functionality to generate text embeddings for paper abstracts
and store them in a vector database with paper metadata.

The module uses the text-embedding-qwen3-embedding-4b model via a local LM Studio API
and stores the embeddings in ChromaDB for efficient similarity search.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import requests
import chromadb
from chromadb.config import Settings

from .config import get_config

logger = logging.getLogger(__name__)


class EmbeddingsError(Exception):
    """Exception raised for embedding operations."""

    pass


class EmbeddingsManager:
    """
    Manager for generating and storing text embeddings.

    This class handles:
    - Connecting to LM Studio API for embedding generation
    - Creating and managing a ChromaDB collection
    - Embedding paper abstracts with metadata
    - Similarity search operations

    Parameters
    ----------
    lm_studio_url : str, optional
        URL of the LM Studio API endpoint, by default "http://localhost:1234"
    model_name : str, optional
        Name of the embedding model, by default "text-embedding-qwen3-embedding-4b"
    chroma_path : str or Path, optional
        Path to the ChromaDB persistent storage, by default "./chroma_db"
    collection_name : str, optional
        Name of the ChromaDB collection, by default "neurips_papers"

    Attributes
    ----------
    lm_studio_url : str
        LM Studio API endpoint URL.
    model_name : str
        Embedding model name.
    chroma_path : Path
        Path to ChromaDB storage.
    collection_name : str
        ChromaDB collection name.
    client : chromadb.Client or None
        ChromaDB client instance.
    collection : chromadb.Collection or None
        Active ChromaDB collection.

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.connect()
    >>> em.create_collection()
    >>> embedding = em.generate_embedding("This is a paper abstract.")
    >>> em.add_paper(paper_id=1, abstract="...", metadata={...})
    >>> results = em.search_similar("machine learning", n_results=5)
    >>> em.close()
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        model_name: Optional[str] = None,
        chroma_path: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the EmbeddingsManager.

        Parameters are optional and will use values from environment/config if not provided.

        Parameters
        ----------
        lm_studio_url : str, optional
            URL of the LM Studio API endpoint. If None, uses config value.
        model_name : str, optional
            Name of the embedding model. If None, uses config value.
        chroma_path : str or Path, optional
            Path to the ChromaDB persistent storage. If None, uses config value.
        collection_name : str, optional
            Name of the ChromaDB collection. If None, uses config value.
        """
        config = get_config()
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.model_name = model_name or config.embedding_model
        self.chroma_path = Path(chroma_path or config.embedding_db_path)
        self.collection_name = collection_name or config.collection_name
        self.client: Optional[Any] = None  # chromadb.Client
        self.collection: Optional[Any] = None  # chromadb.Collection

    def connect(self) -> None:
        """
        Connect to ChromaDB.

        Creates the storage directory if it doesn't exist.

        Raises
        ------
        EmbeddingsError
            If connection fails.
        """
        try:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Connected to ChromaDB at: {self.chroma_path}")
        except Exception as e:
            raise EmbeddingsError(f"Failed to connect to ChromaDB: {str(e)}") from e

    def close(self) -> None:
        """
        Close the ChromaDB connection.

        Does nothing if not connected.
        """
        if self.client:
            self.client = None
            self.collection = None
            logger.info("ChromaDB connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def test_lm_studio_connection(self) -> bool:
        """
        Test connection to LM Studio API.

        Returns
        -------
        bool
            True if connection is successful, False otherwise.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> if em.test_lm_studio_connection():
        ...     print("LM Studio is accessible")
        """
        try:
            # Try to get models list
            response = requests.get(f"{self.lm_studio_url}/v1/models", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to LM Studio at {self.lm_studio_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to LM Studio: {str(e)}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using LM Studio API.

        Parameters
        ----------
        text : str
            Text to generate embedding for.

        Returns
        -------
        List[float]
            Embedding vector.

        Raises
        ------
        EmbeddingsError
            If embedding generation fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> embedding = em.generate_embedding("Sample text")
        >>> len(embedding)
        4096
        """
        if not text or not text.strip():
            raise EmbeddingsError("Cannot generate embedding for empty text")

        try:
            response = requests.post(
                f"{self.lm_studio_url}/v1/embeddings",
                json={"model": self.model_name, "input": text},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "data" not in data or len(data["data"]) == 0:
                raise EmbeddingsError("No embedding data in API response")

            embedding = data["data"][0]["embedding"]
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except requests.exceptions.RequestException as e:
            raise EmbeddingsError(f"Failed to generate embedding via LM Studio: {str(e)}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise EmbeddingsError(f"Invalid response format from LM Studio: {str(e)}") from e

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
        >>> em.connect()
        >>> em.create_collection()
        >>> em.create_collection(reset=True)  # Reset existing collection
        """
        if not self.client:
            raise EmbeddingsError("Not connected to ChromaDB")

        try:
            if reset:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass  # Collection might not exist

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "NeurIPS paper abstracts and metadata"},
            )
            logger.info(f"Created/retrieved collection: {self.collection_name}")

        except Exception as e:
            raise EmbeddingsError(f"Failed to create collection: {str(e)}") from e

    def add_paper(
        self,
        paper_id: Union[int, str],
        abstract: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Add a paper to the vector database.

        Parameters
        ----------
        paper_id : int or str
            Unique identifier for the paper.
        abstract : str
            Paper abstract text.
        metadata : dict, optional
            Additional metadata to store with the embedding.
        embedding : List[float], optional
            Pre-computed embedding. If None, will generate from abstract.

        Raises
        ------
        EmbeddingsError
            If adding paper fails or collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.connect()
        >>> em.create_collection()
        >>> em.add_paper(
        ...     paper_id=1,
        ...     abstract="This paper presents...",
        ...     metadata={"title": "Example Paper", "year": 2025}
        ... )
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        if not abstract or not abstract.strip():
            logger.warning(f"Skipping paper {paper_id}: empty abstract")
            return

        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = self.generate_embedding(abstract)

            # Prepare metadata - convert all values to strings for ChromaDB compatibility
            meta = metadata or {}
            meta = {k: str(v) if v is not None else "" for k, v in meta.items()}

            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[abstract],
                metadatas=[meta],
                ids=[str(paper_id)],
            )
            logger.debug(f"Added paper {paper_id} to collection")

        except Exception as e:
            raise EmbeddingsError(f"Failed to add paper {paper_id}: {str(e)}") from e

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar papers using semantic similarity.

        Parameters
        ----------
        query : str
            Query text to search for.
        n_results : int, optional
            Number of results to return, by default 10
        where : dict, optional
            Metadata filter conditions.

        Returns
        -------
        dict
            Search results containing ids, distances, documents, and metadatas.

        Raises
        ------
        EmbeddingsError
            If search fails or collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.connect()
        >>> em.create_collection()
        >>> results = em.search_similar("deep learning transformers", n_results=5)
        >>> for i, paper_id in enumerate(results['ids'][0]):
        ...     print(f"{i+1}. Paper {paper_id}: {results['metadatas'][0][i]}")
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")

        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )

            logger.info(f"Found {len(results['ids'][0])} similar papers")
            return dict(results)  # type: ignore[arg-type]

        except Exception as e:
            raise EmbeddingsError(f"Failed to search: {str(e)}") from e

    def get_collection_stats(self) -> Dict[str, Any]:
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
        >>> em.connect()
        >>> em.create_collection()
        >>> stats = em.get_collection_stats()
        >>> print(f"Collection has {stats['count']} papers")
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata,
            }
        except Exception as e:
            raise EmbeddingsError(f"Failed to get collection stats: {str(e)}") from e

    def embed_from_database(
        self,
        db_path: Union[str, Path],
        where_clause: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Embed papers from a SQLite database.

        Reads papers from the database and generates embeddings for their abstracts.

        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database file.
        where_clause : str, optional
            SQL WHERE clause to filter papers (e.g., "decision = 'Accept'")
        progress_callback : callable, optional
            Callback function to report progress. Called with (current, total) number of papers processed.

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
        >>> em.connect()
        >>> em.create_collection()
        >>> count = em.embed_from_database("neurips.db")
        >>> print(f"Embedded {count} papers")
        >>> # Only embed accepted papers
        >>> count = em.embed_from_database("neurips.db", where_clause="decision = 'Accept'")
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            db_path = Path(db_path)
            if not db_path.exists():
                raise EmbeddingsError(f"Database not found: {db_path}")

            # Connect to database
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check which columns exist in the papers table
            cursor.execute("PRAGMA table_info(papers)")
            columns = {row[1] for row in cursor.fetchall()}

            # Build query with lightweight schema fields
            base_columns = ["uid", "title", "abstract", "authors", "keywords", "session"]

            query = f"SELECT {', '.join(base_columns)} FROM papers"
            if where_clause:
                query += f" WHERE {where_clause}"

            cursor.execute(query)
            rows = cursor.fetchall()
            total = len(rows)

            logger.info(f"Found {total} papers to embed")

            if total == 0:
                conn.close()
                return 0

            # Process papers one by one
            embedded_count = 0
            for i, row in enumerate(rows):
                paper_id = row["uid"]
                abstract = row["abstract"]

                if not abstract or not abstract.strip():
                    logger.warning(f"Skipping paper {paper_id}: no abstract")
                    continue

                metadata = {
                    "title": row["title"] or "",
                    "authors": row["authors"] or "",
                    "keywords": row["keywords"] or "",
                    "session": row["session"] or "",
                }

                try:
                    self.add_paper(paper_id, abstract, metadata)
                    embedded_count += 1

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, total)

                except Exception as e:
                    logger.error(f"Failed to embed paper {paper_id}: {str(e)}")
                    continue

            conn.close()
            logger.info(f"Successfully embedded {embedded_count} papers")
            return embedded_count

        except sqlite3.Error as e:
            raise EmbeddingsError(f"Database error: {str(e)}") from e
        except Exception as e:
            raise EmbeddingsError(f"Failed to embed from database: {str(e)}") from e
