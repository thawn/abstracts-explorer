"""
This module provides functionality to generate text embeddings for paper abstracts
and store them in a vector database.

The module uses an OpenAI-compatible API (such as LM Studio or blablador) to generate
embeddings and stores them in ChromaDB for efficient similarity search.  ChromaDB
stores only paper UIDs, vectors, and a SHA-256 hash of the embedded text; all paper
metadata lives in the SQL database.
"""

import hashlib
import logging
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx2
from openai import OpenAI
import chromadb
from chromadb.config import Settings

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager, normalize_model_name

logger = logging.getLogger(__name__)

_TOKEN_SAFETY_MARGIN_RATIO = 0.1
_CHAR_FALLBACK_RATIO = 2
_MAX_CONTEXT_RETRIES = 3
_CONTEXT_RETRY_SHRINK = 0.75


@lru_cache(maxsize=1)
def _get_token_encoding() -> Optional[Any]:
    """Return a cached tiktoken encoding, or ``None`` when unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.debug("tiktoken unavailable (%s); falling back to character truncation", exc)
        return None


def _count_tokens(text: str) -> Optional[int]:
    """Count tokens with the cached encoding, returning ``None`` without one."""
    encoding = _get_token_encoding()
    if encoding is None:
        return None
    return len(encoding.encode(text))


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens."""
    if max_tokens <= 0 or not text:
        return text

    encoding = _get_token_encoding()
    if encoding is None:
        return text[: max_tokens * _CHAR_FALLBACK_RATIO]

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def _is_context_length_error(error: Exception) -> bool:
    """Return ``True`` when *error* reports an exceeded input context window."""
    message = str(error).lower()
    markers = (
        "maximum context length",
        "context window",
        "context length",
        "too many tokens",
        "reduce the length of the input",
    )
    return any(marker in message for marker in markers)


def _context_length_limit_from_error(error: Exception) -> Optional[int]:
    """Extract the model's maximum context length from an API error, if present."""
    match = re.search(r"maximum context length is (\d+)", str(error), re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


HASH_PREFIX = "sha256:"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def hash_text(text: str) -> str:
    """
    Return the content hash used to identify embedded text.

    Parameters
    ----------
    text : str
        Text that was embedded.

    Returns
    -------
    str
        ``sha256:`` followed by the hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def is_content_hash(value: Optional[str]) -> bool:
    """
    Return whether *value* is a content hash produced by :func:`hash_text`.

    Parameters
    ----------
    value : str or None
        Candidate ChromaDB document value.

    Returns
    -------
    bool
        ``True`` when *value* matches the ``sha256:<hex>`` format.
    """
    return bool(value) and bool(_HASH_RE.match(str(value)))


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
    max_input_tokens : int, optional
        Maximum number of tokens sent to the embedding model per request. Longer
        inputs are truncated before the API call. If None, uses the value from
        config (default: 8192). Set to 0 to disable proactive truncation.

    Attributes
    ----------
    lm_studio_url : str
        OpenAI-compatible API endpoint URL.
    model_name : str
        Embedding model name.
    max_input_tokens : int
        Configured maximum number of input tokens per embedding request.
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
        max_input_tokens: Optional[int] = None,
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
        max_input_tokens : int, optional
            Maximum number of tokens sent to the embedding model per request.
            If None, uses the value from config (default: 8192).
            Set to 0 to disable proactive truncation.
        """
        config = get_config()
        self.lm_studio_url = (lm_studio_url or config.llm_backend_url).rstrip("/")
        self.llm_backend_auth_token = auth_token or config.llm_backend_auth_token
        self.model_name = model_name or config.embedding_model
        self.max_input_tokens = (
            max_input_tokens if max_input_tokens is not None else config.embedding_max_input_tokens
        )
        self._learned_token_budget: Optional[int] = None

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

    def _effective_token_budget(self) -> int:
        """
        Return the token budget used before sending text to the API.

        The configured model limit is reduced by a safety margin to account for
        tokenizer differences. A smaller budget learned from a previous API
        context-length error takes precedence.

        Returns
        -------
        int
            Token budget, or 0 when proactive truncation is disabled.
        """
        limit = self.max_input_tokens
        if limit > 0:
            margin = max(1, int(limit * _TOKEN_SAFETY_MARGIN_RATIO))
            budget = max(1, limit - margin)
        else:
            budget = 0

        if self._learned_token_budget is not None:
            budget = self._learned_token_budget if budget <= 0 else min(budget, self._learned_token_budget)

        return budget

    def _prepare_embedding_input(self, text: str) -> str:
        """
        Truncate *text* to the effective token budget.

        Parameters
        ----------
        text : str
            Text that will be sent to the embedding API.

        Returns
        -------
        str
            Original text when it fits, otherwise a truncated prefix.
        """
        budget = self._effective_token_budget()
        if budget <= 0:
            return text

        truncated = _truncate_to_token_limit(text, budget)
        if truncated == text:
            return text

        original_tokens = _count_tokens(text)
        truncated_tokens = _count_tokens(truncated)
        if original_tokens is not None and truncated_tokens is not None:
            logger.warning(
                "Truncated embedding input from %d to %d tokens (budget %d) for model %s",
                original_tokens,
                truncated_tokens,
                budget,
                self.model_name,
            )
        else:
            logger.warning(
                "Truncated embedding input from %d to %d characters for model %s",
                len(text),
                len(truncated),
                self.model_name,
            )
        return truncated

    def _shrink_embedding_input(self, original_text: str, current_text: str, error: Exception) -> str:
        """
        Produce a shorter input after the backend rejects a context-length overflow.

        Parameters
        ----------
        original_text : str
            Full, untruncated text.
        current_text : str
            Input that the backend just rejected.
        error : Exception
            Error raised by the embedding API.

        Returns
        -------
        str
            Shortened input to retry.
        """
        current_tokens = _count_tokens(current_text)
        if current_tokens is None:
            return current_text[: max(1, len(current_text) // 2)]

        target = max(1, int(current_tokens * _CONTEXT_RETRY_SHRINK))
        parsed_limit = _context_length_limit_from_error(error)
        if parsed_limit is not None:
            margin = max(1, int(parsed_limit * _TOKEN_SAFETY_MARGIN_RATIO))
            target = min(target, max(1, parsed_limit - margin))
        target = min(target, max(1, current_tokens - 1))

        self._learned_token_budget = target
        return _truncate_to_token_limit(original_text, target)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using OpenAI-compatible API.

        Rate limiting (if configured via ``requests_per_minute``) is handled
        transparently by the underlying ``httpx2`` transport.

        Inputs longer than the model's context window are truncated before the
        request. If the backend still reports an overflow, the input is shrunk
        and retried a few times.

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

        prepared_text = self._prepare_embedding_input(text)

        for attempt in range(_MAX_CONTEXT_RETRIES + 1):
            try:
                response = self.openai_client.embeddings.create(model=self.model_name, input=prepared_text)

                if not response.data or len(response.data) == 0:
                    raise EmbeddingsError("No embedding data in API response")

                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding with dimension: {len(embedding)}")
                return embedding

            except Exception as e:
                if not _is_context_length_error(e) or attempt >= _MAX_CONTEXT_RETRIES:
                    raise EmbeddingsError(f"Failed to generate embedding via OpenAI API: {str(e)}") from e

                prepared_text = self._shrink_embedding_input(text, prepared_text, e)
                logger.warning(
                    "Embedding input exceeded the model context window; retrying with a shorter input "
                    "(attempt %d/%d)",
                    attempt + 1,
                    _MAX_CONTEXT_RETRIES,
                )

        raise EmbeddingsError("Failed to generate embedding via OpenAI API: retries exhausted")

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
            result = self.collection.get(ids=[paper_id], include=[])
            # If the result has any IDs, the paper exists
            return len(result["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking if paper {paper_id} exists: {str(e)}")
            return False

    def paper_needs_update(self, paper: dict) -> bool:
        """
        Check if a paper's embedding needs to be updated.

        The stored document is a SHA-256 content hash of the text that was
        embedded.  A paper needs an update when the hash for its current title
        and abstract differs from the stored hash.

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
        """
        try:
            existing_paper = self.collection.get(ids=[paper["uid"]], include=["documents"])
            if not existing_paper or len(existing_paper["ids"]) == 0:
                return True

            existing_documents = existing_paper.get("documents", [])
            if not existing_documents or not existing_documents[0]:
                return True

            return existing_documents[0] != self.embedding_hash_from_paper(paper)

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

    @classmethod
    def embedding_hash_from_paper(cls, paper: dict) -> str:
        """
        Return the content hash for a paper's embedding text.

        Parameters
        ----------
        paper : dict
            Dictionary containing at least ``title`` and ``abstract``.

        Returns
        -------
        str
            ``sha256:<hex>`` hash of :meth:`embedding_text_from_paper`.
        """
        return hash_text(cls.embedding_text_from_paper(paper))

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
            embedding = self.generate_embedding(embedding_text)

            self.collection.add(
                embeddings=[embedding],
                documents=[hash_text(embedding_text)],
                ids=[paper["uid"]],
            )
            logger.debug(f"Added paper {paper['uid']} to collection")

        except Exception as e:
            raise EmbeddingsError(f"Failed to add paper {paper['uid']}: {str(e)}") from e

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar papers using semantic similarity.

        Parameters
        ----------
        query : str
            Query text to search for.
        n_results : int, optional
            Number of results to return, by default 10.
        ids : list of str, optional
            Restrict the search to these paper UIDs.

        Returns
        -------
        dict
            ChromaDB query results containing ``ids`` and ``distances``.

        Raises
        ------
        EmbeddingsError
            If search fails or collection not initialized.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.init()
        >>> results = em.search_similar("deep learning transformers", n_results=5)
        >>> for i, paper_id in enumerate(results['ids'][0]):
        ...     print(f"{i+1}. Paper {paper_id}: {results['distances'][0][i]}")
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")
        if ids is not None and not ids:
            return {"ids": [[]], "distances": [[]]}

        try:
            query_embedding = self.generate_embedding(query)
            if ids is not None:
                n_results = min(n_results, len(ids))

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                ids=ids,
                include=["distances"],
            )

            logger.info(f"Found {len(results['ids'][0])} similar papers")
            return {"ids": results.get("ids", [[]]), "distances": results.get("distances", [[]])}

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

    def _query_collection(
        self,
        query: str,
        ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run a distance query against the embeddings collection.

        Parameters
        ----------
        query : str
            The search query text.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        query_embedding : list[float], optional
            Pre-computed embedding for the query.  When omitted, an embedding
            is generated with :meth:`generate_embedding`.

        Returns
        -------
        dict
            Dictionary with ``query_embedding``, ``ids``, ``distances``, and
            ``total_considered``.

        Raises
        ------
        EmbeddingsError
            If the query is empty, the collection is empty, no IDs matched, or
            no results are returned.
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")

        if ids is not None and not ids:
            raise EmbeddingsError("No papers matched the filters")

        if query_embedding is None:
            query_embedding = self.generate_embedding(query)

        total_count = self.collection.count()
        if total_count == 0:
            raise EmbeddingsError("No papers in collection")

        n_results_query = min(_MAX_QUERY_RESULTS, len(ids) if ids is not None else total_count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_query,
            include=["distances"],
            ids=ids,
        )

        paper_ids = results["ids"][0] if results.get("ids") else []
        distances = results["distances"][0] if results.get("distances") else []
        if not paper_ids:
            raise EmbeddingsError("No results from collection query")

        return {
            "query_embedding": query_embedding,
            "ids": paper_ids,
            "distances": distances,
            "total_considered": len(paper_ids),
        }

    @staticmethod
    def _matching_pairs(
        paper_ids: List[str],
        distances: List[float],
        distance_threshold: float,
    ) -> List[Tuple[str, float]]:
        """
        Return identifier/distance pairs within a distance threshold.

        Parameters
        ----------
        paper_ids : list[str]
            Paper identifiers, sorted by distance.
        distances : list[float]
            Distances corresponding to ``paper_ids``.
        distance_threshold : float
            Maximum distance to include.

        Returns
        -------
        list of tuple
            ``(paper_id, distance)`` pairs within the threshold.
        """
        pairs: List[Tuple[str, float]] = []
        for paper_id, distance in zip(paper_ids, distances):
            if distance > distance_threshold:
                break
            pairs.append((paper_id, float(distance)))
        return pairs

    def count_papers_within_distance(
        self,
        query: str,
        distance_threshold: float = 1.1,
        ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> int:
        """
        Count papers within a distance threshold.

        Parameters
        ----------
        query : str
            The search query text.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        ids : list of str, optional
            Restrict the count to these paper UIDs.
        query_embedding : list[float], optional
            Pre-computed embedding for the query.

        Returns
        -------
        int
            Number of papers within the distance threshold.

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails.
        """
        try:
            result = self._query_collection(
                query=query,
                ids=ids,
                query_embedding=query_embedding,
            )
            matches = self._matching_pairs(result["ids"], result["distances"], distance_threshold)
            return len(matches)
        except EmbeddingsError:
            raise
        except Exception as e:
            logger.error(f"Error counting papers within distance: {e}", exc_info=True)
            raise EmbeddingsError(f"Failed to count papers within distance: {str(e)}") from e

    def query_ids_within_distance(
        self,
        query: str,
        distance_threshold: float = 1.1,
        ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
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
        query_embedding : list[float], optional
            Pre-computed embedding for the query.

        Returns
        -------
        dict
            Dictionary containing ``query``, ``query_embedding``, ``distance``,
            ``ids``, ``distances``, ``count``, and ``total_considered``.

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails.
        """
        try:
            result = self._query_collection(
                query=query,
                ids=ids,
                query_embedding=query_embedding,
            )
            matches = self._matching_pairs(result["ids"], result["distances"], distance_threshold)
            return {
                "query": query,
                "query_embedding": result["query_embedding"],
                "distance": distance_threshold,
                "ids": [paper_id for paper_id, _ in matches],
                "distances": [distance for _, distance in matches],
                "count": len(matches),
                "total_considered": result["total_considered"],
            }
        except EmbeddingsError:
            raise
        except Exception as e:
            logger.error(f"Error querying papers within distance: {e}", exc_info=True)
            raise EmbeddingsError(f"Failed to query papers within distance: {str(e)}") from e

    def delete_embeddings_by_ids(self, ids: List[str]) -> int:
        """
        Delete embeddings by paper UID.

        Parameters
        ----------
        ids : list of str
            Paper UIDs to remove.

        Returns
        -------
        int
            Number of embeddings deleted.

        Raises
        ------
        EmbeddingsError
            If the deletion fails.
        """
        if not ids:
            return 0

        try:
            deleted = 0
            for start in range(0, len(ids), 1000):
                batch = ids[start : start + 1000]
                self.collection.delete(ids=batch)
                deleted += len(batch)
            return deleted
        except Exception as e:
            raise EmbeddingsError(f"Failed to delete embeddings: {str(e)}") from e

    # ------------------------------------------------------------------
    # Registry export / import helpers
    # ------------------------------------------------------------------

    def export_embeddings(self, ids: List[str]) -> Dict[str, Any]:
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
            If the export fails or the collection has not been compacted.
        """
        try:
            exported_ids: List[str] = []
            documents: List[str] = []
            embeddings: List[List[float]] = []

            for start in range(0, len(ids), 1000):
                batch = ids[start : start + 1000]
                results = self.collection.get(ids=batch, include=["documents", "embeddings"])
                result_ids = results.get("ids", [])
                result_docs = results.get("documents", [])
                result_embeddings = results.get("embeddings", [])

                by_id = {uid: (result_docs[index], result_embeddings[index]) for index, uid in enumerate(result_ids)}
                for uid in batch:
                    if uid not in by_id:
                        continue
                    document, embedding = by_id[uid]
                    if not is_content_hash(document):
                        raise EmbeddingsError(
                            "Collection contains uncompacted documents. "
                            "Run 'abstracts-explorer embeddings compact' before exporting."
                        )
                    exported_ids.append(uid)
                    documents.append(str(document))
                    vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                    embeddings.append(vector)

            return {
                "format_version": 1,
                "ids": exported_ids,
                "documents": documents,
                "embeddings": embeddings,
            }
        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to export embeddings: {str(e)}") from e

    def import_embeddings(
        self,
        data: Dict[str, Any],
        replace_ids: Optional[List[str]] = None,
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
        if data.get("format_version") != 1:
            raise EmbeddingsError("Unsupported embeddings artifact format")

        ids = data.get("ids", [])
        documents = data.get("documents", [])
        embeddings = data.get("embeddings", [])
        if not ids:
            return 0
        if not (len(ids) == len(documents) == len(embeddings)):
            raise EmbeddingsError("Embeddings artifact fields have inconsistent lengths")
        if not all(is_content_hash(document) for document in documents):
            raise EmbeddingsError("Embeddings artifact contains invalid content hashes")

        try:
            if replace_ids:
                self.delete_embeddings_by_ids(replace_ids)

            imported = 0
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start : start + batch_size]
                self.collection.add(
                    ids=batch_ids,
                    documents=documents[start : start + batch_size],
                    embeddings=embeddings[start : start + batch_size],
                )
                imported += len(batch_ids)
            return imported
        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to import embeddings: {str(e)}") from e

    # ------------------------------------------------------------------
    # Collection compaction
    # ------------------------------------------------------------------

    def remove_all_metadata(self, batch_size: int = 1000) -> int:
        """
        Remove all per-paper metadata from the collection.

        Parameters
        ----------
        batch_size : int, optional
            Number of records to process per batch.

        Returns
        -------
        int
            Number of collection records processed.

        Raises
        ------
        EmbeddingsError
            If the update fails.
        """
        try:
            all_ids = list(self.collection.get(include=[])["ids"])
            processed = 0
            for start in range(0, len(all_ids), batch_size):
                batch = all_ids[start : start + batch_size]
                existing = self.collection.get(ids=batch, include=["metadatas"])
                keys: set[str] = set()
                for metadata in existing.get("metadatas") or []:
                    if metadata:
                        keys.update(metadata)
                if not keys:
                    processed += len(batch)
                    continue
                null_metadata = {key: None for key in keys}
                self.collection.update(ids=batch, metadatas=[null_metadata] * len(batch))
                processed += len(batch)
            return processed
        except Exception as e:
            raise EmbeddingsError(f"Failed to remove metadata: {str(e)}") from e

    def _update_documents(self, ids: List[str], documents: List[str]) -> None:
        """Update documents without touching embeddings or metadata."""
        self.collection._client._update(  # type: ignore[attr-defined]
            collection_id=self.collection.id,
            ids=ids,
            embeddings=None,
            metadatas=None,
            documents=documents,
            tenant=self.collection.tenant,
            database=self.collection.database,
        )

    def hash_all_documents(self, batch_size: int = 1000) -> int:
        """
        Replace stored full-text documents with content hashes.

        Parameters
        ----------
        batch_size : int, optional
            Number of records to process per batch.

        Returns
        -------
        int
            Number of documents updated.

        Raises
        ------
        EmbeddingsError
            If the update fails.
        """
        try:
            all_ids = list(self.collection.get(include=[])["ids"])
            updated = 0
            for start in range(0, len(all_ids), batch_size):
                batch = all_ids[start : start + batch_size]
                existing = self.collection.get(ids=batch, include=["documents"])
                result_ids = existing.get("ids", [])
                result_documents = existing.get("documents", [])
                update_ids: List[str] = []
                update_documents: List[str] = []
                for uid, document in zip(result_ids, result_documents):
                    if not document:
                        continue
                    new_document = str(document) if is_content_hash(document) else hash_text(str(document))
                    if new_document == document:
                        continue
                    update_ids.append(uid)
                    update_documents.append(new_document)
                if update_ids:
                    self._update_documents(update_ids, update_documents)
                    updated += len(update_ids)
            return updated
        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Failed to hash documents: {str(e)}") from e

    def prune_orphan_embeddings(self, database: DatabaseManager, batch_size: int = 1000) -> int:
        """
        Delete embeddings whose UIDs no longer exist in the paper database.

        Parameters
        ----------
        database : DatabaseManager
            Database used to resolve valid paper UIDs.
        batch_size : int, optional
            Number of records to process per batch.

        Returns
        -------
        int
            Number of orphan embeddings deleted.

        Raises
        ------
        EmbeddingsError
            If the operation fails.
        """
        try:
            embedding_ids = set(self.collection.get(include=[])["ids"])
            paper_ids = set(database.get_paper_uids())
            orphans = list(embedding_ids - paper_ids)
            return self.delete_embeddings_by_ids(orphans)
        except Exception as e:
            raise EmbeddingsError(f"Failed to prune orphan embeddings: {str(e)}") from e
