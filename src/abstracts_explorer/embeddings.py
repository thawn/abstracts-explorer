"""
Embeddings Module
=================

This module provides functionality to generate text embeddings for paper abstracts
and store them in a vector database with paper metadata.

The module uses an OpenAI-compatible API (such as LM Studio or blablador) to generate
embeddings and stores them in ChromaDB for efficient similarity search.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from openai import OpenAI
import chromadb
from chromadb.config import Settings

from .config import get_config
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class EmbeddingsError(Exception):
    """Exception raised for embedding operations."""

    pass


class EmbeddingsManager:
    """
    Manager for generating and storing text embeddings.

    This class handles:
    - Connecting to OpenAI-compatible API for embedding generation
    - Creating and managing a ChromaDB collection
    - Embedding paper abstracts with metadata
    - Similarity search operations

    Parameters
    ----------
    lm_studio_url : str, optional
        URL of the OpenAI-compatible API endpoint, by default "http://localhost:1234"
    model_name : str, optional
        Name of the embedding model, by default "text-embedding-qwen3-embedding-4b"
    chroma_path : str or Path, optional
        Path to the ChromaDB persistent storage, by default "./chroma_db"
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers"

    Attributes
    ----------
    lm_studio_url : str
        OpenAI-compatible API endpoint URL.
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
    >>> em.add_paper(paper_dict)
    >>> results = em.search_similar("machine learning", n_results=5)
    >>> em.close()
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        auth_token: Optional[str] = None,
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
            URL of the OpenAI-compatible API endpoint. If None, uses config value.
        model_name : str, optional
            Name of the embedding model. If None, uses config value.
        chroma_path : str or Path, optional
            Path to the ChromaDB persistent storage. If None, uses config value.
        collection_name : str, optional
            Name of the ChromaDB collection. If None, uses config value.
        """
        config = get_config()
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.llm_backend_auth_token = auth_token or config.llm_backend_auth_token
        self.model_name = model_name or config.embedding_model
        self.chroma_path = Path(chroma_path or config.embedding_db_path)
        self.collection_name = collection_name or config.collection_name
        self.client: Optional[Any] = None  # chromadb.Client
        self.collection: Optional[Any] = None  # chromadb.Collection
        
        # OpenAI client - lazy loaded on first use to avoid API calls during test collection
        self._openai_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> OpenAI:
        """
        Get the OpenAI client, creating it lazily on first access.
        
        This lazy loading prevents API calls during test collection.
        
        Returns
        -------
        OpenAI
            Initialized OpenAI client instance.
        """
        if self._openai_client is None:
            self._openai_client = OpenAI(
                base_url=f"{self.lm_studio_url}/v1",
                api_key=self.llm_backend_auth_token or "lm-studio-local"
            )
        return self._openai_client

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
        Test connection to OpenAI-compatible API endpoint.

        Returns
        -------
        bool
            True if connection is successful, False otherwise.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> if em.test_lm_studio_connection():
        ...     print("API is accessible")
        """
        try:
            # Try to get models list
            _ = self.openai_client.models.list()
            logger.info(f"Successfully connected to OpenAI API at {self.lm_studio_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to OpenAI API: {str(e)}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using OpenAI-compatible API.

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
            response = self.openai_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            if not response.data or len(response.data) == 0:
                raise EmbeddingsError("No embedding data in API response")

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            raise EmbeddingsError(f"Failed to generate embedding via OpenAI API: {str(e)}") from e

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

    def paper_exists(self, paper_id: str) -> bool:
        """
        Check if a paper already exists in the collection.

        Parameters
        ----------
        paper_id : int or str
            Unique identifier for the paper.

        Returns
        -------
        bool
            True if paper exists in collection, False otherwise.

        Raises
        ------
        EmbeddingsError
            If collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.connect()
        >>> em.create_collection()
        >>> em.paper_exists("uid1")
        False
        >>> em.add_paper(paper_dict)
        >>> em.paper_exists("uid1")
        True
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            # Try to get the paper by ID
            result = self.collection.get(ids=[paper_id])
            # If the result has any IDs, the paper exists
            return len(result["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking if paper {paper_id} exists: {str(e)}")
            return False

    def paper_needs_update(self, paper: dict) -> bool:
        """
        Check if a paper needs to be updated in the collection.

        Parameters
        ----------
        paper : dict
            Dictionary containing paper information.

        Returns
        -------
        bool
            True if the paper needs to be updated, False otherwise.

        Raises
        ------
        EmbeddingsError
            If collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.connect()
        >>> em.create_collection()
        >>> em.paper_needs_update({"id": 1, "abstract": "Updated abstract"})
        True
        >>> em.paper_needs_update({"id": 1, "abstract": "This paper presents..."})
        False
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            existing_paper = self.collection.get(ids=[paper["uid"]])
            if not existing_paper or len(existing_paper["ids"]) == 0:
                return True  # Paper does not exist, needs to be added

            # Compare existing embedding text with new paper data
            existing_documents = existing_paper.get("documents", [])
            if not existing_documents:
                return True  # No document stored, needs update

            existing_embedding_text = existing_documents[0]
            new_embedding_text = self.embedding_text_from_paper(paper)
            return existing_embedding_text != new_embedding_text

        except Exception as e:
            logger.warning(f"Error checking if paper {paper['uid']} needs update: {str(e)}")
            return False

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

    def add_paper(self, paper: dict) -> None:
        """
        Add a paper to the vector database.

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
        >>> em.connect()
        >>> em.create_collection()
        >>> em.add_paper(paper_dict)
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            embedding_text = self.embedding_text_from_paper(paper)
            # Generate embedding if not provided
            embedding = self.generate_embedding(embedding_text)

            # Prepare metadata - convert all values to strings for ChromaDB compatibility
            meta = paper.copy()
            meta = {k: str(v) if v is not None else "" for k, v in meta.items()}

            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[embedding_text],
                metadatas=[meta],
                ids=[paper["uid"]],
            )
            logger.debug(f"Added paper {paper['uid']} to collection")

        except Exception as e:
            raise EmbeddingsError(f"Failed to add paper {paper['uid']}: {str(e)}") from e

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
        >>> results = em.search_similar("deep learning transformers", n_results=5, where={"year": 2025})
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

    def check_model_compatibility(self, db_path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the current embedding model matches the one stored in the database.

        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database file.

        Returns
        -------
        tuple of (bool, str or None, str or None)
            - compatible: True if models match or no model is stored, False if they differ
            - stored_model: Name of the model stored in the database, or None if not set
            - current_model: Name of the current model

        Raises
        ------
        EmbeddingsError
            If database operations fail.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> compatible, stored, current = em.check_model_compatibility("neurips.db")
        >>> if not compatible:
        ...     print(f"Model mismatch: stored={stored}, current={current}")
        """
        try:
            db_path = Path(db_path)
            if not db_path.exists():
                # If database doesn't exist, consider it compatible (no previous embeddings)
                return True, None, self.model_name

            # Use DatabaseManager to check the stored model
            db_manager = DatabaseManager(db_path)
            db_manager.connect()

            stored_model = db_manager.get_embedding_model()
            db_manager.close()

            # If no model is stored, consider it compatible (first time embedding)
            if stored_model is None:
                return True, None, self.model_name

            # Check if models match
            compatible = stored_model == self.model_name
            return compatible, stored_model, self.model_name

        except Exception as e:
            raise EmbeddingsError(f"Failed to check model compatibility: {str(e)}") from e

    def embed_from_database(
        self,
        db_path: Union[str, Path],
        where_clause: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_recreate: bool = False,
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

            # Use DatabaseManager for database operations
            db_manager = DatabaseManager(db_path)
            db_manager.connect()

            # Store the embedding model in the database
            db_manager.set_embedding_model(self.model_name)

            query = "SELECT * FROM papers"
            if where_clause:
                query += f" WHERE {where_clause}"

            rows = db_manager.query(query)
            total = len(rows)

            logger.info(f"Found {total} papers to embed")

            if total == 0:
                db_manager.close()
                return 0

            # Process papers one by one
            embedded_count = 0
            skipped_count = 0
            for i, row in enumerate(rows):
                # Convert sqlite3.Row to dict
                paper = dict(row)

                # Check if paper already exists in the collection and if it needs to be updated
                # Skip this check if force_recreate is True
                if not force_recreate and not self.paper_needs_update(paper):
                    logger.debug(f"Skipping paper {paper['uid']}: already exists in collection")
                    skipped_count += 1
                    # Still call progress callback to update the progress bar
                    if progress_callback:
                        progress_callback(i + 1, total)
                    continue
                else:
                    try:
                        self.add_paper(paper)
                        embedded_count += 1
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(i + 1, total)

                    except Exception as e:
                        logger.error(f"Failed to embed paper {paper['uid']}: {str(e)}")
                        continue

            db_manager.close()
            logger.info(f"Successfully embedded {embedded_count} papers, skipped {skipped_count} existing papers")
            return embedded_count

        except Exception as e:
            raise EmbeddingsError(f"Failed to embed from database: {str(e)}") from e
