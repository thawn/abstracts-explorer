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
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers"

    Attributes
    ----------
    lm_studio_url : str
        OpenAI-compatible API endpoint URL.
    model_name : str
        Embedding model name.
    embedding_db : str
        ChromaDB configuration - URL for HTTP service or path for local storage.
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
        collection_name : str, optional
            Name of the ChromaDB collection. If None, uses config value.
        """
        config = get_config()
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.llm_backend_auth_token = auth_token or config.llm_backend_auth_token
        self.model_name = model_name or config.embedding_model
        
        # Get ChromaDB configuration from config
        self.embedding_db = config.embedding_db
        
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

        Uses HTTP client if embedding_db is a URL, otherwise uses persistent client
        with local storage directory.

        Raises
        ------
        EmbeddingsError
            If connection fails.
        """
        try:
            if self.embedding_db.startswith("http://") or self.embedding_db.startswith("https://"):
                # Use HTTP client for remote ChromaDB service
                # Parse URL properly using urllib
                parsed = urlparse(self.embedding_db)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8000
                
                self.client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.debug(f"Connected to ChromaDB HTTP service at: {self.embedding_db}")
            else:
                # Use persistent client for local storage
                chroma_path = Path(self.embedding_db)
                chroma_path.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=str(chroma_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.debug(f"Connected to ChromaDB at: {chroma_path}")
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
            logger.debug("ChromaDB connection closed")

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
            logger.debug(f"Successfully connected to OpenAI API at {self.lm_studio_url}")
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
            logger.debug(f"Created/retrieved collection: {self.collection_name}")

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

    def check_model_compatibility(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the current embedding model matches the one stored in the database.

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
        >>> compatible, stored, current = em.check_model_compatibility()
        >>> if not compatible:
        ...     print(f"Model mismatch: stored={stored}, current={current}")
        """
        try:
            # Use DatabaseManager to check the stored model
            db_manager = DatabaseManager()
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
        where_clause: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_recreate: bool = False,
    ) -> int:
        """
        Embed papers from the database.

        Reads papers from the database and generates embeddings for their abstracts.

        Parameters
        ----------
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
        >>> count = em.embed_from_database()
        >>> print(f"Embedded {count} papers")
        >>> # Only embed accepted papers
        >>> count = em.embed_from_database(where_clause="decision = 'Accept'")
        """
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")

        try:
            # Use DatabaseManager for database operations
            db_manager = DatabaseManager()
            db_manager.connect()

            # Store the embedding model in the database
            db_manager.set_embedding_model(self.model_name)

            query = "SELECT * FROM papers"
            if where_clause:
                query += f" WHERE {where_clause}"

            rows = db_manager.query(query)
            total = len(rows)

            logger.debug(f"Found {total} papers to embed")

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

    def search_papers_semantic(
        self,
        query: str,
        database,
        limit: int = 10,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search for papers using embeddings.
        
        This function combines embedding-based similarity search with metadata filtering
        and retrieves complete paper information from the database.
        
        Parameters
        ----------
        query : str
            Search query text
        database : DatabaseManager
            Database manager for retrieving full paper details
        limit : int, optional
            Maximum number of results to return, by default 10
        sessions : list of str, optional
            Filter by paper sessions
        years : list of int, optional
            Filter by publication years
        conferences : list of str, optional
            Filter by conference names
            
        Returns
        -------
        list of dict
            List of paper dictionaries with complete information
            
        Raises
        ------
        EmbeddingsError
            If search fails
            
        Examples
        --------
        >>> papers = em.search_papers_semantic(
        ...     "transformers in vision",
        ...     database=db,
        ...     limit=5,
        ...     years=[2024, 2025]
        ... )
        """
        from .paper_utils import format_search_results, PaperFormattingError
        
        # Build metadata filter for embeddings search
        # NOTE: All metadata is stored as strings in ChromaDB (see add_paper method, line 445)
        # so we must convert filter values to strings for matching
        filter_conditions: List[Dict[str, Any]] = []
        if sessions:
            filter_conditions.append({"session": {"$in": sessions}})
        if years:
            # Convert years to strings to match ChromaDB metadata storage format
            year_strs: List[str] = [str(y) for y in years]
            filter_conditions.append({"year": {"$in": year_strs}})
        if conferences:
            filter_conditions.append({"conference": {"$in": conferences}})
        
        # Use $and operator if multiple conditions, otherwise use single condition
        where_filter: Optional[Dict[str, Any]] = None
        if len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        
        logger.info(f"Semantic search - query: {query}, filter: sessions={sessions}, years={years}, conferences={conferences}")
        logger.info(f"Where filter: {where_filter}")
        
        # Get more results initially to account for filtering
        results = self.search_similar(query, n_results=limit * 2, where=where_filter)
        
        logger.info(f"Search results count: {len(results.get('ids', [[]])[0]) if results else 0}")
        
        # Transform ChromaDB results to paper format using shared utility
        try:
            papers = format_search_results(results, database, include_documents=False)
        except PaperFormattingError:
            # No valid papers found
            return []
        
        # Limit results (filtering already done at database level)
        return papers[:limit]

    def find_papers_within_distance(
        self,
        database,
        query: str,
        distance_threshold: float = 1.1,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Find papers within a specified distance from a custom search query.
        
        This method treats the search query as a clustering center and returns
        papers within the specified Euclidean distance radius in embedding space.
        
        Parameters
        ----------
        database : DatabaseManager
            Database manager instance for retrieving paper details
        query : str
            The search query text
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1
        conferences : list[str], optional
            Filter results to only include papers from these conferences
        years : list[int], optional
            Filter results to only include papers from these years
        
        Returns
        -------
        dict
            Dictionary containing:
            - query: str - The search query
            - query_embedding: list[float] - The generated embedding for the query
            - distance: float - The distance threshold used
            - papers: list[dict] - Papers within the distance radius with their distances
            - count: int - Number of papers found
        
        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails
        
        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.connect()
        >>> em.create_collection()
        >>> db = DatabaseManager()
        >>> db.connect()
        >>> results = em.find_papers_within_distance(db, "machine learning", 1.1)
        >>> print(f"Found {results['count']} papers")
        >>> 
        >>> # With filters
        >>> results = em.find_papers_within_distance(
        ...     db, "deep learning", 1.1, 
        ...     conferences=["NeurIPS"], 
        ...     years=[2023, 2024]
        ... )
        """
        from abstracts_explorer.paper_utils import get_paper_with_authors, PaperFormattingError
        
        if not self.collection:
            raise EmbeddingsError("Collection not initialized. Call create_collection() first.")
        
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")
        
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Get total count of papers in collection
            total_count = self.collection.count()
            if total_count == 0:
                raise EmbeddingsError("No papers in collection")
            
            # Build where clause for filtering
            where_clause: Optional[Dict[str, Any]] = None
            if conferences or years:
                filters: list[Dict[str, Any]] = []
                if conferences:
                    if len(conferences) == 1:
                        filters.append({"conference": conferences[0]})
                    else:
                        filters.append({"conference": {"$in": conferences}})
                
                if years:
                    if len(years) == 1:
                        filters.append({"year": years[0]})
                    else:
                        filters.append({"year": {"$in": years}})
                
                # Combine filters with $and if multiple
                if len(filters) == 1:
                    where_clause = filters[0]
                else:
                    where_clause = {"$and": filters}
            
            # Query all papers and get distances
            # Using collection.query() which returns papers sorted by distance
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=total_count,  # Get all papers
                include=["distances", "metadatas"],
                where=where_clause
            )
            
            # Extract results (query returns nested lists)
            paper_ids = results['ids'][0] if results.get('ids') else []
            distances = results['distances'][0] if results.get('distances') else []
            
            if not paper_ids:
                raise EmbeddingsError("No results from collection query")
            
            # Filter papers within distance threshold
            matching_papers = []
            for idx, (paper_id, distance) in enumerate(zip(paper_ids, distances)):
                if distance <= distance_threshold:
                    # Get full paper details from database using uid
                    try:
                        paper_dict = get_paper_with_authors(database, paper_id)
                        paper_dict["distance"] = float(distance)
                        matching_papers.append(paper_dict)
                    except PaperFormattingError:
                        # Paper not found in database, skip it
                        logger.warning(f"Paper {paper_id} not found in database, skipping")
                        continue
                else:
                    # Since results are sorted by distance, we can break early
                    break
            
            return {
                "query": query,
                "query_embedding": query_embedding,
                "distance": distance_threshold,
                "papers": matching_papers,
                "count": len(matching_papers),
            }
            
        except EmbeddingsError:
            # Re-raise EmbeddingsError as-is
            raise
        except Exception as e:
            logger.error(f"Error finding papers within distance: {e}", exc_info=True)
            raise EmbeddingsError(f"Failed to find papers within distance: {str(e)}") from e
