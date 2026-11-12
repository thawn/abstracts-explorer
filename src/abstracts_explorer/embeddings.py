"""
This module provides functionality to generate text embeddings for paper abstracts
and store them in a vector database with paper metadata.

The module uses an OpenAI-compatible API (such as LM Studio or blablador) to generate
embeddings and stores them in ChromaDB for efficient similarity search.
"""

import logging
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx2
from openai import OpenAI
import chromadb
from chromadb.config import Settings

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager, normalize_model_name

logger = logging.getLogger(__name__)


def _contains_metadata_key(value: Any, key: str) -> bool:
    """Recursively check whether *key* appears in a ChromaDB where clause."""
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_contains_metadata_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_metadata_key(item, key) for item in value)
    return False


def build_chromadb_where_clause(
    conferences: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
    sessions: Optional[Sequence[str]] = None,
    base_where: Optional[Dict[str, Any]] = None,
    combine_with: str = "$and",
) -> Optional[Dict[str, Any]]:
    """
    Build a ChromaDB ``where`` clause from metadata filters.

    The helper centralizes all ChromaDB metadata-filter composition.  It
    converts years to strings (ChromaDB stores metadata as strings), combines
    multiple conditions with ``$and`` (or ``$or``), and merges an optional
    user-supplied base clause without duplicating existing
    conference/year/session filters.

    Parameters
    ----------
    conferences : sequence of str, optional
        Conference names to filter by.
    years : sequence of int, optional
        Publication years to filter by.
    sessions : sequence of str, optional
        Session names to filter by.
    base_where : dict, optional
        Existing ChromaDB where clause to merge with the filters.
    combine_with : str, optional
        Logical operator used to combine conditions, either ``"$and"``
        (default) or ``"$or"``.

    Returns
    -------
    dict or None
        A new where clause, or ``None`` when no filters were supplied.

    Raises
    ------
    ValueError
        If *base_where* is not a dict, *combine_with* is invalid, or the
        base clause contains an invalid logical-operator value.

    Examples
    --------
    >>> build_chromadb_where_clause(["NeurIPS"], [2024])
    {'$and': [{'conference': {'$in': ['NeurIPS']}}, {'year': {'$in': ['2024']}}]}
    >>> build_chromadb_where_clause(base_where={"session": "Oral"})
    {'session': 'Oral'}
    """
    if base_where is not None and not isinstance(base_where, dict):
        raise ValueError(f"WHERE clause must be a dict, got {type(base_where).__name__}")
    if combine_with not in ("$and", "$or"):
        raise ValueError(f"combine_with must be '$and' or '$or', got {combine_with!r}")

    where = deepcopy(base_where) if base_where else None

    conditions: List[Dict[str, Any]] = []
    if conferences and not _contains_metadata_key(where, "conference"):
        conference_values = [conferences] if isinstance(conferences, str) else list(conferences)
        conditions.append({"conference": {"$in": conference_values}})
    if years and not _contains_metadata_key(where, "year"):
        year_values = [years] if isinstance(years, (str, int)) else list(years)
        conditions.append({"year": {"$in": [str(year) for year in year_values]}})
    if sessions and not _contains_metadata_key(where, "session"):
        session_values = [sessions] if isinstance(sessions, str) else list(sessions)
        conditions.append({"session": {"$in": session_values}})

    if not conditions:
        return where

    if where is None:
        return conditions[0] if len(conditions) == 1 else {combine_with: conditions}

    existing_combined = where.get(combine_with)
    if existing_combined is not None:
        if not isinstance(existing_combined, list):
            raise ValueError(f"WHERE clause '{combine_with}' value must be a list")
        existing_combined.extend(conditions)
        return where

    return {combine_with: [where, *conditions]}


class RateLimitedTransport(httpx2.BaseTransport, httpx2.AsyncBaseTransport):
    """
    An httpx2 transport that enforces a maximum requests-per-minute rate.

    Supports both synchronous and asynchronous clients.  Wraps an existing
    transport and acquires tokens from a shared token bucket rate limiter before
    forwarding requests.  This provides thread-safe, global rate limiting that
    works correctly under Waitress's multi-threaded WSGI server as well as with
    Pydantic AI's async OpenAI provider.

    Parameters
    ----------
    transport : httpx2.BaseTransport or httpx2.AsyncBaseTransport
        The underlying transport to delegate requests to.
    requests_per_minute : int
        Maximum number of requests per minute. Must be > 0.
    """

    def __init__(
        self,
        transport: httpx2.BaseTransport | httpx2.AsyncBaseTransport,
        requests_per_minute: int,
    ) -> None:
        from abstracts_explorer.rate_limiter import (
            TokenBucketRateLimiter,
            get_global_rate_limiter,
            set_global_rate_limiter,
        )

        self._transport = transport
        self._rpm = requests_per_minute

        global_rate_limiter = get_global_rate_limiter()
        if global_rate_limiter is None:
            global_rate_limiter = TokenBucketRateLimiter(requests_per_minute=requests_per_minute)
            set_global_rate_limiter(global_rate_limiter)

        self._rate_limiter = global_rate_limiter

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* after acquiring a rate-limit token."""
        if not isinstance(self._transport, httpx2.BaseTransport):
            raise TypeError("Cannot use a synchronous client with an async transport.")
        self._rate_limiter.acquire(tokens=1.0)
        response = self._transport.handle_request(request)
        return response

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* after acquiring a rate-limit token asynchronously."""
        if not isinstance(self._transport, httpx2.AsyncBaseTransport):
            raise TypeError("Cannot use an async client with a synchronous transport.")
        await self._rate_limiter.async_acquire(tokens=1.0)
        response = await self._transport.handle_async_request(request)
        return response

    def close(self) -> None:
        """Close the underlying sync transport."""
        if isinstance(self._transport, httpx2.BaseTransport):
            self._transport.close()

    async def aclose(self) -> None:
        """Close the underlying async transport."""
        if isinstance(self._transport, httpx2.AsyncBaseTransport):
            await self._transport.aclose()


class EmbeddingsError(Exception):
    """Exception raised for embedding operations."""

    pass


# Maximum number of results to request from ChromaDB in a single query.
# Prevents "too many SQL variables" errors in the underlying SQLite backend.
_MAX_QUERY_RESULTS = 32766  # this is the maximum for sqlite 3.32 and above


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
    requests_per_minute : int, optional
        Maximum number of API requests per minute. Set to 0 to disable rate limiting.
        If None, uses the value from config (default: 60).

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
    client : chromadb.Client
        ChromaDB client instance. Connected automatically on first access.
    collection : chromadb.Collection
        Active ChromaDB collection. Created automatically on first access
        (which also connects the client if not yet connected).

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.add_paper(paper_dict)  # connect() and create_collection() called automatically
    >>> results = em.search_similar("machine learning", n_results=5)
    >>> em.close()
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
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
        requests_per_minute : int, optional
            Maximum number of API requests per minute. Set to 0 to disable rate limiting.
            If None, uses the value from config (default: 60).
        """
        config = get_config()
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.llm_backend_auth_token = auth_token or config.llm_backend_auth_token
        self.model_name = model_name or config.embedding_model

        # Get ChromaDB configuration from config
        self.embedding_db = config.embedding_db

        self.collection_name = collection_name or config.collection_name
        self._client: Optional[Any] = None  # chromadb.Client
        self._collection: Optional[Any] = None  # chromadb.Collection
        self._init_lock = threading.RLock()

        # OpenAI client - lazy loaded on first use to avoid API calls during test collection
        self._openai_client: Optional[OpenAI] = None

        # Rate limiting: maximum API requests per minute (0 = unlimited)
        self.requests_per_minute = (
            requests_per_minute if requests_per_minute is not None else config.requests_per_minute
        )

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
                    self.connect()
        assert self._client is not None  # connect() always sets _client or raises EmbeddingsError
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    @property
    def collection(self) -> Any:
        """
        Get the ChromaDB collection, creating it automatically on first access.

        Calling this property for the first time also triggers :meth:`connect` if
        the client has not been initialized yet.

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

    @property
    def openai_client(self) -> OpenAI:
        """
        Get the OpenAI client, creating it lazily on first access.

        When ``requests_per_minute`` is greater than 0 a :class:`RateLimitedTransport`
        is wrapped around the default httpx2 transport and passed as the ``http_client``
        argument so that every HTTP request is automatically throttled.

        This lazy loading prevents API calls during test collection.

        Returns
        -------
        OpenAI
            Initialized OpenAI client instance.
        """
        if self._openai_client is None:
            http_client: Optional[Any] = None
            if self.requests_per_minute > 0:
                transport = RateLimitedTransport(httpx2.HTTPTransport(), self.requests_per_minute)
                http_client = httpx2.Client(transport=transport)
            self._openai_client = OpenAI(
                base_url=f"{self.lm_studio_url}/v1",
                api_key=self.llm_backend_auth_token or "lm-studio-local",
                http_client=http_client,
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

                self._client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.debug(f"Connected to ChromaDB HTTP service at: {self.embedding_db}")
            else:
                # Use persistent client for local storage
                chroma_path = Path(self.embedding_db)
                chroma_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
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
        if self._client:
            self._client = None
            self._collection = None
            logger.debug("ChromaDB connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def init(self, reset: bool = False) -> None:
        """
        Connect to ChromaDB and create or retrieve the collection.

        Parameters
        ----------
        reset : bool, optional
            If True, delete the existing collection before creating it,
            by default False.

        Raises
        ------
        EmbeddingsError
            If connecting to ChromaDB or creating the collection fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.init()
        """
        with self._init_lock:
            self.connect()
            self.create_collection(reset=reset)

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

    def verbose_api_connection_test(self, model_hint: Optional[str] = None) -> bool:
        """
        Test the OpenAI-compatible API connection and print user-facing messages.

        Parameters
        ----------
        model_hint : str, optional
            Description of the expected model shown in the failure message.
            If None, a message using ``self.model_name`` is shown.

        Returns
        -------
        bool
            True if the connection is successful, False otherwise.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> if em.verbose_api_connection_test():
        ...     print("API is accessible")
        """
        print("\n🔌 Testing OpenAI API connection...")
        if not self.test_lm_studio_connection():
            print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
            print("\nPlease ensure:", file=sys.stderr)
            print(f"  - OpenAI-compatible API is running at {self.lm_studio_url}", file=sys.stderr)
            hint = model_hint or f"The {self.model_name} model is loaded"
            print(f"  - {hint}", file=sys.stderr)
            return False
        print("✅ Successfully connected to OpenAI API")
        return True

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using OpenAI-compatible API.

        Rate limiting (if configured via ``requests_per_minute``) is handled
        transparently by the underlying ``httpx2`` transport.

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
            response = self.openai_client.embeddings.create(model=self.model_name, input=text)

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
        >>> em.init(reset=True)  # Reset an existing collection
        """
        try:
            if reset:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass  # Collection might not exist

            self._collection = self.client.get_or_create_collection(
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
        >>> em.init()
        >>> em.paper_exists("uid1")
        False
        >>> em.add_paper(paper_dict)
        >>> em.paper_exists("uid1")
        True
        """
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
        >>> em.init()
        >>> em.paper_needs_update({"id": 1, "abstract": "Updated abstract"})
        True
        >>> em.paper_needs_update({"id": 1, "abstract": "This paper presents..."})
        False
        """
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
            return True

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

    @staticmethod
    def parse_chromadb_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a raw ChromaDB metadata dict through the LightweightPaper model.

        ChromaDB stores all values as strings (see :meth:`add_paper`).  This
        method converts a raw metadata dict into one with properly typed values
        by running it through :func:`prepare_chroma_db_paper_data` and then
        validating via :class:`LightweightPaper`.

        Parameters
        ----------
        metadata : dict
            Raw metadata dictionary from ChromaDB.

        Returns
        -------
        dict
            Metadata dictionary with values converted to their canonical types.
            Authors will be a ``list[str]`` and keywords a ``list[str]``.

        Examples
        --------
        >>> raw = {"title": "My Paper", "year": "2024", "original_id": "42",
        ...        "authors": "Alice;Bob", "abstract": "An abstract",
        ...        "session": "ML", "poster_position": "1",
        ...        "conference": "NeurIPS"}
        >>> parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        >>> parsed["year"]
        2024
        >>> parsed["authors"]
        ['Alice', 'Bob']

        See Also
        --------
        LightweightPaper : Pydantic model used for validation.
        prepare_chroma_db_paper_data : Converts ChromaDB string fields to
            proper types before validation.
        """
        from abstracts_explorer.plugin import prepare_chroma_db_paper_data, LightweightPaper

        prepared = prepare_chroma_db_paper_data(metadata.copy())
        return LightweightPaper(**prepared).model_dump(exclude_none=True)

    @staticmethod
    def _serialize_metadata_for_chromadb(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a metadata dict to ChromaDB-compatible string values.

        ChromaDB only accepts ``str``, ``int``, ``float``, ``bool``, or ``None``
        as metadata values.  :meth:`export_embeddings` (and the registry export
        path) runs raw ChromaDB metadata through :meth:`parse_chromadb_metadata`
        which converts the semicolon-separated *authors* string and the
        comma-separated *keywords* string back to Python lists.  When that data
        is round-tripped through JSON and then passed back to ChromaDB via
        :meth:`import_embeddings`, the list values must be re-serialised.

        List fields use the same helpers as
        :func:`~abstracts_explorer.plugin.serialize_authors_to_string` and
        :func:`~abstracts_explorer.plugin.serialize_keywords_to_string` to
        keep the stored format consistent with the SQL database.  All other
        values are converted to strings so that ChromaDB metadata filters work
        reliably (e.g. ``{"year": "2025"}``).

        Parameters
        ----------
        metadata : dict
            Metadata dict that may contain list values.

        Returns
        -------
        dict
            Metadata dict with all values converted to ChromaDB-compatible types.
        """
        from abstracts_explorer.plugin import serialize_authors_to_string, serialize_keywords_to_string

        result: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                result[k] = ""
            elif isinstance(v, list):
                if k == "authors":
                    result[k] = serialize_authors_to_string(v)
                elif k == "keywords":
                    result[k] = serialize_keywords_to_string(v)
                else:
                    result[k] = str(v)
            else:
                result[k] = str(v)
        return result

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
        >>> em.init()
        >>> em.add_paper(paper_dict)
        """
        try:
            embedding_text = self.embedding_text_from_paper(paper)
            # Generate embedding if not provided
            embedding = self.generate_embedding(embedding_text)

            # Prepare metadata - serialize all values for ChromaDB compatibility,
            # using the same format as _serialize_metadata_for_chromadb so that
            # add_paper and import_embeddings produce identical stored representations.
            meta = self._serialize_metadata_for_chromadb(paper.copy())

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
        ids: Optional[List[str]] = None,
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
        >>> em.init()
        >>> results = em.search_similar("deep learning transformers", n_results=5, where={"year": 2025})
        >>> for i, paper_id in enumerate(results['ids'][0]):
        ...     print(f"{i+1}. Paper {paper_id}: {results['metadatas'][0][i]}")
        """
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
                ids=ids,
            )

            logger.info(f"Found {len(results['ids'][0])} similar papers")

            # Parse metadata through LightweightPaper model to convert
            # string values back to their proper types (e.g. year → int).
            parsed = dict(results)  # type: ignore[arg-type]
            if parsed.get("metadatas"):
                parsed["metadatas"] = [
                    [self.parse_chromadb_metadata(m) for m in batch] for batch in parsed["metadatas"]
                ]
            return parsed

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
        >>> em.init()
        >>> stats = em.get_collection_stats()
        >>> print(f"Collection has {stats['count']} papers")
        """
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

            stored_model = db_manager.get_embedding_model()
            db_manager.close()

            # If no model is stored, consider it compatible (first time embedding)
            if stored_model is None:
                return True, None, self.model_name

            # Check if models match
            compatible = normalize_model_name(stored_model) == normalize_model_name(self.model_name)
            return compatible, stored_model, self.model_name

        except Exception as e:
            raise EmbeddingsError(f"Failed to check model compatibility: {str(e)}") from e

    def embed_from_database(
        self,
        where_filters: Optional[Dict[str, Any]] = None,
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
        >>> em.init()
        >>> count = em.embed_from_database()
        >>> print(f"Embedded {count} papers")
        >>> # Only embed accepted papers
        >>> count = em.embed_from_database(where_filters={"award": {"$ne": None}})
        """
        try:
            # Use DatabaseManager for database operations
            db_manager = DatabaseManager()

            # Store the embedding model in the database
            db_manager.set_embedding_model(self.model_name)

            rows = db_manager.get_paper(where_filters, limit=0)
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

    def count_papers_within_distance(
        self,
        database,
        query: str,
        distance_threshold: float = 1.1,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
    ) -> int:
        """
        Count papers within a distance threshold.

        Delegates to :meth:`find_papers_within_distance` and returns only the
        count of matching papers.

        Parameters
        ----------
        database : DatabaseManager
            Database manager instance for retrieving paper details.
        query : str
            The search query text.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        conferences : list[str], optional
            Filter results to only include papers from these conferences.
        years : list[int], optional
            Filter results to only include papers from these years.

        Returns
        -------
        int
            Number of papers within the distance threshold.

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails.
        """
        result = self.find_papers_within_distance(
            database=database,
            query=query,
            distance_threshold=distance_threshold,
            conferences=conferences,
            years=years,
            max_papers=0,
        )
        return result["count"]

    def find_papers_within_distance(
        self,
        database,
        query: str,
        distance_threshold: float = 1.1,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        query_embedding: Optional[List[float]] = None,
        max_papers: Optional[int] = None,
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
        query_embedding : list[float], optional
            Pre-computed embedding for the query.  When provided, the
            embedding generation step is skipped, which avoids redundant
            LLM API calls when calling this method repeatedly with the
            same query (e.g. topic-evolution analysis).
        max_papers : int, optional
            Maximum number of paper details to return.  When ``None``
            (default), all matching papers are fetched from the database.
            ``count`` always reflects the total number of matching papers,
            even when ``papers`` is capped.  Use ``0`` to skip the database
            lookup entirely and only obtain the count.

        Returns
        -------
        dict
            Dictionary containing:

            - query: str - The search query
            - query_embedding: list[float] - The generated embedding for the query
            - distance: float - The distance threshold used
            - papers: list[dict] - Papers within the distance radius with their distances
              (capped to ``max_papers`` when provided)
            - count: int - Number of papers found within the distance threshold
            - total_considered: int - Total number of papers matching the
              conference/year filters (before distance filtering)

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.init()
        >>> db = DatabaseManager()
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
        from abstracts_explorer.paper_utils import PaperFormattingError

        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")

        try:
            # Use the pre-computed embedding if provided, otherwise generate one
            if query_embedding is None:
                query_embedding = self.generate_embedding(query)

            # Get total count of papers in collection
            total_count = self.collection.count()
            if total_count == 0:
                raise EmbeddingsError("No papers in collection")

            # Build where clause for filtering
            # NOTE: All metadata is stored as strings in ChromaDB (see add_paper method),
            # so we must convert filter values to strings for matching.
            where_clause = build_chromadb_where_clause(conferences, years)

            # Query papers and get distances.
            # Cap n_results to avoid ChromaDB / SQLite "too many SQL variables"
            # errors that occur when the collection is large (SQLite has a
            # default limit of 32,766 bound parameters).
            n_results_query = min(total_count, _MAX_QUERY_RESULTS)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_query,
                include=["distances"],
                where=where_clause,
            )

            # Extract results (query returns nested lists)
            paper_ids = results["ids"][0] if results.get("ids") else []
            distances = results["distances"][0] if results.get("distances") else []

            if not paper_ids:
                raise EmbeddingsError("No results from collection query")

            # Collect matching (paper_id, distance) pairs.  Results are sorted
            # by distance, so we can stop at the first paper outside the radius.
            matching: List[Tuple[str, float]] = []
            for paper_id, distance in zip(paper_ids, distances):
                if distance <= distance_threshold:
                    matching.append((paper_id, float(distance)))
                else:
                    break

            count = len(matching)
            if max_papers is None:
                selected = matching
            else:
                # Python slicing is safe when max_papers exceeds len(matching).
                # Clamp negative values so they don't select from the end.
                selected = matching[: max(0, max_papers)]

            # Fetch paper details in batches to avoid one database round-trip
            # per matching paper (a major bottleneck for large collections).
            matching_papers: List[Dict[str, Any]] = []
            if selected:
                selected_ids = [paper_id for paper_id, _ in selected]
                distance_by_id = dict(selected)
                fetched_papers: List[Dict[str, Any]] = []
                batch_size = 500
                for start in range(0, len(selected_ids), batch_size):
                    batch = selected_ids[start : start + batch_size]
                    try:
                        fetched_papers.extend(database.get_papers_by_uids(batch))
                    except PaperFormattingError:
                        logger.warning("Some papers were not found in the database, skipping")
                papers_by_id = {paper["uid"]: paper for paper in fetched_papers if paper}

                for paper_id in selected_ids:
                    paper_dict = papers_by_id.get(paper_id)
                    if paper_dict is None:
                        logger.warning(f"Paper {paper_id} not found in database, skipping")
                        continue
                    paper_dict["distance"] = distance_by_id[paper_id]
                    matching_papers.append(paper_dict)

            return {
                "query": query,
                "query_embedding": query_embedding,
                "distance": distance_threshold,
                "papers": matching_papers,
                "count": count,
                "total_considered": len(paper_ids),
            }

        except EmbeddingsError:
            # Re-raise EmbeddingsError as-is
            raise
        except Exception as e:
            logger.error(f"Error finding papers within distance: {e}", exc_info=True)
            raise EmbeddingsError(f"Failed to find papers within distance: {str(e)}") from e

    def delete_embeddings_by_filter(
        self,
        conference: Optional[str] = None,
        year: Optional[int] = None,
    ) -> int:
        """
        Delete embeddings that match the given conference and/or year filter.

        Only embeddings whose metadata matches **all** supplied criteria are
        removed.  At least one of *conference* or *year* must be provided;
        calling this method with both set to ``None`` raises ``ValueError`` to
        prevent accidental deletion of the entire collection.

        Parameters
        ----------
        conference : str, optional
            Conference name to match (exact, case-sensitive, as stored in
            ChromaDB metadata).
        year : int, optional
            Publication year to match.

        Returns
        -------
        int
            Number of embeddings deleted.

        Raises
        ------
        ValueError
            If both *conference* and *year* are ``None``.
        EmbeddingsError
            If the deletion fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.init()
        >>> deleted = em.delete_embeddings_by_filter(conference="NeurIPS", year=2024)
        >>> print(f"Deleted {deleted} embeddings")
        """
        if conference is None and year is None:
            raise ValueError("At least one of 'conference' or 'year' must be provided.")

        try:
            where_filter = build_chromadb_where_clause(
                conferences=[conference] if conference is not None else None,
                years=[year] if year is not None else None,
            )
            assert where_filter is not None

            existing = self.collection.get(where=where_filter)
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} embeddings matching filter {where_filter}")
            return len(ids_to_delete)

        except (ValueError, EmbeddingsError):
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to delete embeddings by filter: {str(e)}") from e

    # ------------------------------------------------------------------
    # Registry export / import helpers
    # ------------------------------------------------------------------

    def export_embeddings(
        self,
        conference: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Export embeddings for a given conference and year to a JSON-serializable dict.

        Parameters
        ----------
        conference : str
            Conference name to export.
        year : int
            Year to export.

        Returns
        -------
        dict
            Dictionary containing ``ids``, ``documents``, ``metadatas``, and
            ``embeddings`` lists.  Embedding vectors are converted to plain
            Python lists so the returned dict is always JSON-serializable.

        Raises
        ------
        EmbeddingsError
            If the export fails.
        """
        try:
            results = self.collection.get(
                include=["documents", "embeddings", "metadatas"],
                where=build_chromadb_where_clause(
                    conferences=[conference],
                    years=[year],
                ),
            )
            embeddings = results.get("embeddings", [])
            # ChromaDB may return embeddings as numpy ndarrays; convert to plain lists
            # so the dict is always JSON-serializable.
            if embeddings is not None:
                embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
            # Parse metadata through LightweightPaper model to convert
            # string values back to their proper types (e.g. year → int).
            raw_metadatas = results.get("metadatas", [])
            parsed_metadatas = [self.parse_chromadb_metadata(m) for m in raw_metadatas]
            return {
                "ids": results.get("ids", []),
                "documents": results.get("documents", []),
                "metadatas": parsed_metadatas,
                "embeddings": embeddings,
            }
        except Exception as e:
            raise EmbeddingsError(f"Failed to export embeddings: {str(e)}") from e

    def import_embeddings(
        self,
        data: Dict[str, Any],
        conference: str,
        year: int,
        batch_size: int = 100,
    ) -> int:
        """
        Import embeddings for a given conference and year from a dictionary.

        Existing embeddings for the same conference and year are **deleted**
        before importing (replace semantics).

        Parameters
        ----------
        data : dict
            Dictionary with ``ids``, ``documents``, ``metadatas``, and
            ``embeddings`` lists (as returned by :meth:`export_embeddings`).
        conference : str
            Conference name being imported.
        year : int
            Year being imported.
        batch_size : int
            Number of embeddings to add per batch.

        Returns
        -------
        int
            Number of embeddings imported.

        Raises
        ------
        EmbeddingsError
            If the import fails.
        """
        try:
            # Remove existing embeddings for this conference+year
            try:
                existing = self.collection.get(
                    where=build_chromadb_where_clause(
                        conferences=[conference],
                        years=[year],
                    ),
                )
                if existing["ids"]:
                    self.collection.delete(ids=existing["ids"])
                    logger.info(f"Deleted {len(existing['ids'])} existing embeddings " f"for {conference}/{year}")
            except Exception:
                logger.debug("No existing embeddings to delete")

            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            embeddings = data.get("embeddings", [])

            if not ids:
                return 0

            imported = 0
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                add_kwargs: Dict[str, Any] = {"ids": batch_ids}
                if documents:
                    add_kwargs["documents"] = documents[i : i + batch_size]
                if metadatas:
                    add_kwargs["metadatas"] = [
                        self._serialize_metadata_for_chromadb(m) for m in metadatas[i : i + batch_size]
                    ]
                if embeddings:
                    add_kwargs["embeddings"] = embeddings[i : i + batch_size]

                self.collection.add(**add_kwargs)
                imported += len(batch_ids)

            return imported

        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to import embeddings: {str(e)}") from e

    def update_paper_metadata(self, updates: Dict[str, Dict[str, Any]]) -> int:
        """
        Update metadata fields for existing papers without changing their embeddings.

        Fetches the current metadata for each UID, merges the supplied field
        updates, re-serialises the result and calls ``collection.update``.
        Papers whose UIDs are not found in the collection are silently skipped.

        Parameters
        ----------
        updates : dict
            Mapping of paper UID → dict of metadata field → new value.
            Only the keys present in each inner dict are modified; all other
            metadata fields are preserved.

        Returns
        -------
        int
            Number of papers whose metadata was actually updated.

        Raises
        ------
        EmbeddingsError
            If fetching or updating the collection fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.update_paper_metadata({
        ...     "abc123": {"paper_pdf_url": "https://example.com/paper.pdf"}
        ... })
        1
        """
        if not updates:
            return 0

        try:
            ids = list(updates.keys())
            existing = self.collection.get(ids=ids, include=["metadatas"])

            updated_ids = []
            updated_metadatas = []
            for uid, raw_meta in zip(existing["ids"], existing.get("metadatas") or []):
                # raw_meta contains already-serialised string values from ChromaDB.
                # Merge new values and re-serialise so None → "" and lists are handled.
                merged = dict(raw_meta)
                merged.update(updates[uid])
                updated_ids.append(uid)
                updated_metadatas.append(self._serialize_metadata_for_chromadb(merged))

            if updated_ids:
                self.collection.update(ids=updated_ids, metadatas=updated_metadatas)

            return len(updated_ids)

        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to update paper metadata: {str(e)}") from e
