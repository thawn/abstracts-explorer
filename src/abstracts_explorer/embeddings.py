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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from itertools import takewhile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx2
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager, normalize_model_name
from abstracts_explorer.rate_limiter import RateLimitedTransport

logger = logging.getLogger(__name__)

_TOKEN_SAFETY_MARGIN_RATIO = 0.1
_CHAR_FALLBACK_RATIO = 2
_MAX_CONTEXT_RETRIES = 3
_CONTEXT_RETRY_SHRINK = 0.75

HASH_PREFIX = "sha256:"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Maximum number of results to request from ChromaDB in a single query.
# Prevents "too many SQL variables" errors in the underlying SQLite backend.
_MAX_QUERY_RESULTS = 32766  # this is the maximum for sqlite 3.32 and above

_IO_BATCH_SIZE = 1000
_EMBED_BATCH_SIZE = 32


def _token_margin(limit: int) -> int:
    """Return the safety margin applied to a token limit."""
    return max(1, int(limit * _TOKEN_SAFETY_MARGIN_RATIO))


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


def _truncate_with_counts(text: str, max_tokens: int) -> tuple[str, Optional[int], Optional[int]]:
    """
    Truncate *text* to at most *max_tokens* tokens in a single encode/decode pass.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_tokens : int
        Maximum number of tokens. Non-positive values disable truncation.

    Returns
    -------
    tuple of (str, int or None, int or None)
        The (possibly truncated) text, the original token count, and the
        truncated token count. Token counts are ``None`` when tiktoken is
        unavailable or truncation is disabled.
    """
    if max_tokens <= 0 or not text:
        return text, None, None

    encoding = _get_token_encoding()
    if encoding is None:
        return text[: max_tokens * _CHAR_FALLBACK_RATIO], None, None

    tokens = encoding.encode(text)
    original_count = len(tokens)
    if original_count <= max_tokens:
        return text, original_count, original_count
    return encoding.decode(tokens[:max_tokens]), original_count, max_tokens


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


def _iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive slices of *items* with at most *size* elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


@contextmanager
def _translate_errors(message: str) -> Iterator[None]:
    """Wrap unexpected exceptions in :class:`EmbeddingsError`, passing existing ones through."""
    try:
        yield
    except EmbeddingsError:
        raise
    except Exception as e:
        raise EmbeddingsError(f"{message}: {str(e)}") from e


class EmbeddingsError(Exception):
    """Exception raised for embedding operations."""


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
    backend_url : str, optional
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
    backend_url : str
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
    >>> em.add_paper(paper_dict)  # client and collection created automatically
    >>> results = em.search_similar("machine learning", n_results=5)
    >>> em.close()
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
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
        backend_url : str, optional
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
        self.backend_url = (backend_url or config.llm_backend_url).rstrip("/")
        self.llm_backend_auth_token = auth_token or config.llm_backend_auth_token
        self.model_name = model_name or config.embedding_model
        self.max_input_tokens = (
            max_input_tokens if max_input_tokens is not None else config.embedding_max_input_tokens
        )
        self._learned_token_budget: Optional[int] = None

        self.embedding_db = config.embedding_db
        self.collection_name = collection_name or config.collection_name
        self._client: Optional[Any] = None  # chromadb.Client
        self._collection: Optional[Any] = None  # chromadb.Collection
        self._init_lock = threading.RLock()

        self._openai_client: Optional[OpenAI] = None

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

    @property
    def openai_client(self) -> OpenAI:
        """
        Get the OpenAI client, creating it lazily on first access.

        When ``requests_per_minute`` is greater than 0 a :class:`RateLimitedTransport`
        is wrapped around the default httpx2 transport and passed as the ``http_client``
        argument so that every HTTP request is automatically throttled.

        Returns
        -------
        OpenAI
            Initialized OpenAI client instance.
        """
        if self._openai_client is None:
            with self._init_lock:
                if self._openai_client is None:
                    self._openai_client = self._create_openai_client()
        return self._openai_client

    def _create_openai_client(self) -> OpenAI:
        """Create the OpenAI client, optionally with a rate-limited HTTP transport."""
        http_client: Optional[Any] = None
        if self.requests_per_minute > 0:
            transport = RateLimitedTransport(httpx2.HTTPTransport(), self.requests_per_minute)
            http_client = httpx2.Client(transport=transport)
        return OpenAI(
            base_url=f"{self.backend_url}/v1",
            api_key=self.llm_backend_auth_token or "lm-studio-local",
            http_client=http_client,
        )

    def _connect(self) -> None:
        """Connect to ChromaDB via HTTP for URLs or persistent storage for local paths."""
        with _translate_errors("Failed to connect to ChromaDB"):
            parsed = urlparse(self.embedding_db)
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

    def close(self) -> None:
        """
        Close the OpenAI client and reset the ChromaDB connection.

        Safe to call multiple times.
        """
        if self._openai_client is not None:
            try:
                self._openai_client.close()
            except Exception as e:
                logger.debug("Failed to close OpenAI client: %s", e)
            self._openai_client = None
        if self._client is not None:
            self._client = None
            self._collection = None
            logger.debug("ChromaDB connection closed")

    def __enter__(self) -> "EmbeddingsManager":
        """Context manager entry; connects to ChromaDB."""
        _ = self.client
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def test_api_connection(
        self,
        *,
        verbose: bool = False,
        model_hint: Optional[str] = None,
    ) -> bool:
        """
        Test the connection to the OpenAI-compatible API endpoint.

        Parameters
        ----------
        verbose : bool, optional
            If True, print user-facing status messages, by default False.
        model_hint : str, optional
            Description of the expected model shown in the verbose failure
            message. If None, a message using ``self.model_name`` is shown.

        Returns
        -------
        bool
            True if the connection is successful, False otherwise.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> if em.test_api_connection():
        ...     print("API is accessible")
        """
        try:
            _ = self.openai_client.models.list()
            logger.debug("Successfully connected to OpenAI-compatible API at %s", self.backend_url)
            if verbose:
                print("✅ Successfully connected to OpenAI API")
            return True
        except Exception as e:
            logger.warning("Failed to connect to OpenAI-compatible API: %s", str(e))
            if verbose:
                print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
                print("\nPlease ensure:", file=sys.stderr)
                print(f"  - OpenAI-compatible API is running at {self.backend_url}", file=sys.stderr)
                hint = model_hint or f"The {self.model_name} model is loaded"
                print(f"  - {hint}", file=sys.stderr)
            return False

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
            budget = max(1, limit - _token_margin(limit))
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

        truncated, original_tokens, truncated_tokens = _truncate_with_counts(text, budget)
        if truncated == text:
            return text

        if original_tokens is not None:
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
            target = min(target, max(1, parsed_limit - _token_margin(parsed_limit)))
        target = min(target, max(1, current_tokens - 1))

        self._learned_token_budget = target
        truncated, _, _ = _truncate_with_counts(original_text, target)
        return truncated

    def generate_embedding(self, text: str) -> list[float]:
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
        list of float
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

        attempt = 0
        while True:
            try:
                response = self.openai_client.embeddings.create(model=self.model_name, input=prepared_text)
                if not response.data:
                    raise EmbeddingsError("No embedding data in API response")

                embedding = response.data[0].embedding
                logger.debug("Generated embedding with dimension: %d", len(embedding))
                return embedding
            except EmbeddingsError:
                raise
            except Exception as e:
                if not _is_context_length_error(e) or attempt >= _MAX_CONTEXT_RETRIES:
                    raise EmbeddingsError(f"Failed to generate embedding via OpenAI API: {str(e)}") from e

                prepared_text = self._shrink_embedding_input(text, prepared_text, e)
                attempt += 1
                logger.warning(
                    "Embedding input exceeded the model context window; retrying with a shorter input "
                    "(attempt %d/%d)",
                    attempt,
                    _MAX_CONTEXT_RETRIES,
                )

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for several texts in a single API request.

        Texts are truncated to the effective token budget like
        :meth:`generate_embedding`. If the backend rejects the batch because of
        the context window (or returns an unexpected number of vectors), each
        text is retried individually with :meth:`generate_embedding`.

        Parameters
        ----------
        texts : sequence of str
            Texts to generate embeddings for.

        Returns
        -------
        list of list of float
            Embedding vectors in the same order as *texts*.

        Raises
        ------
        EmbeddingsError
            If a text is empty or embedding generation fails.
        """
        if not texts:
            return []

        for text in texts:
            if not text or not text.strip():
                raise EmbeddingsError("Cannot generate embedding for empty text")

        prepared = [self._prepare_embedding_input(text) for text in texts]

        fallback_reason: Optional[str] = None
        try:
            response = self.openai_client.embeddings.create(model=self.model_name, input=prepared)
            if len(response.data) != len(prepared):
                fallback_reason = "unexpected number of embeddings in API response"
            else:
                return [item.embedding for item in response.data]
        except Exception as e:
            if not _is_context_length_error(e):
                raise EmbeddingsError(f"Failed to generate embeddings via OpenAI API: {str(e)}") from e
            fallback_reason = "input exceeded the model context window"

        logger.warning("Batch embedding failed (%s); retrying texts individually", fallback_reason)
        return [self.generate_embedding(text) for text in texts]

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
        with self._init_lock, _translate_errors("Failed to create collection"):
            if reset:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info("Deleted existing collection: %s", self.collection_name)
                except Exception:
                    pass  # Collection might not exist

            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Paper abstracts and metadata"},
            )
            logger.debug("Created/retrieved collection: %s", self.collection_name)

    def paper_needs_update(self, paper: dict) -> bool:
        """
        Check if a paper's embedding needs to be created or updated.

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

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> em.create_collection()
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

            return existing_documents[0] != hash_text(self.embedding_text_from_paper(paper))
        except Exception as e:
            logger.warning("Error checking if paper %s needs update: %s", paper["uid"], str(e))
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

    def _store_embeddings(self, ids: list[str], documents: list[str], embeddings: list[list[float]]) -> None:
        """Create or replace records in the collection (upsert semantics)."""
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents)

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
        with _translate_errors(f"Failed to add paper {paper['uid']}"):
            embedding_text = self.embedding_text_from_paper(paper)
            embedding = self.generate_embedding(embedding_text)
            self._store_embeddings(
                ids=[paper["uid"]],
                documents=[hash_text(embedding_text)],
                embeddings=[embedding],
            )
            logger.debug("Added paper %s to collection", paper["uid"])

    def _query_embeddings(
        self,
        query: str,
        *,
        ids: Optional[list[str]] = None,
        query_embedding: Optional[list[float]] = None,
        n_results: Optional[int] = None,
    ) -> tuple[list[str], list[float], list[float]]:
        """
        Run a distance query against the collection and return flat results.

        Generates the query embedding when not supplied. The number of results
        is capped by ``n_results`` (when given), the number of candidate IDs,
        the collection size, and :data:`_MAX_QUERY_RESULTS`. When no candidates
        exist, empty result lists are returned without querying.

        Parameters
        ----------
        query : str
            The search query text.
        ids : list of str, optional
            Restrict the query to these paper UIDs.
        query_embedding : list of float, optional
            Pre-computed embedding for the query.
        n_results : int, optional
            Maximum number of results to return.

        Returns
        -------
        tuple of (list of str, list of float, list of float)
            Paper IDs, distances, and the query embedding.

        Raises
        ------
        EmbeddingsError
            If the query is empty or the query itself fails.
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")

        if query_embedding is None:
            query_embedding = self.generate_embedding(query)

        candidate_count = len(ids) if ids is not None else self.collection.count()
        limit = min(
            n_results if n_results is not None else candidate_count,
            candidate_count,
            _MAX_QUERY_RESULTS,
        )
        if limit <= 0:
            return [], [], query_embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["distances"],
            ids=ids,
        )
        paper_ids = list(results["ids"][0]) if results.get("ids") else []
        distances = list(results["distances"][0]) if results.get("distances") else []
        return paper_ids, distances, query_embedding

    def search_similar(
        self,
        query: str,
        n_results: int = 10,
        ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Search for similar papers using semantic similarity.

        Parameters
        ----------
        query : str
            Query text to search for.
        n_results : int, optional
            Number of results to return, by default 10.
        ids : list of str, optional
            Restrict the search to these paper UIDs. An empty list returns an
            empty result.

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
        >>> em.create_collection()
        >>> results = em.search_similar("deep learning transformers", n_results=5)
        >>> for i, paper_id in enumerate(results['ids'][0]):
        ...     print(f"{i+1}. Paper {paper_id}: {results['distances'][0][i]}")
        """
        if not query or not query.strip():
            raise EmbeddingsError("Query cannot be empty")
        if ids is not None and not ids:
            return {"ids": [[]], "distances": [[]]}

        with _translate_errors("Failed to search"):
            paper_ids, distances, _ = self._query_embeddings(query, ids=ids, n_results=n_results)
            logger.info("Found %d similar papers", len(paper_ids))
            return {"ids": [paper_ids], "distances": [distances]}

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
        with _translate_errors("Failed to get collection stats"):
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata,
            }

    def check_model_compatibility(self) -> tuple[bool, Optional[str], Optional[str]]:
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
        with _translate_errors("Failed to check model compatibility"):
            with DatabaseManager() as db_manager:
                stored_model = db_manager.get_embedding_model()

            if stored_model is None:
                return True, None, self.model_name
            return (
                normalize_model_name(stored_model) == normalize_model_name(self.model_name),
                stored_model,
                self.model_name,
            )

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
        with _translate_errors("Failed to embed from database"):
            with DatabaseManager() as db_manager:
                db_manager.set_embedding_model(self.model_name)
                rows = db_manager.get_paper(where_filters, limit=0)
                total = len(rows)
                logger.debug("Found %d papers to embed", total)

                embedded_count = 0
                skipped_count = 0
                processed = 0
                for batch in _iter_batches(rows, _EMBED_BATCH_SIZE):
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
                                    self.add_paper(papers[uid])
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
            existing = self.collection.get(ids=uids, include=["documents"])
        except Exception as e:
            logger.warning("Error checking existing embeddings: %s", e)
            return uids

        stored = dict(zip(existing.get("ids", []), existing.get("documents", [])))
        return [uid for uid in uids if stored.get(uid) != hash_text(texts[uid])]

    def query_ids_within_distance(
        self,
        query: str,
        distance_threshold: float = 1.1,
        ids: Optional[list[str]] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> dict[str, Any]:
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
        query_embedding : list of float, optional
            Pre-computed embedding for the query.

        Returns
        -------
        dict
            Dictionary containing ``query_embedding``, ``ids``, ``distances``,
            and ``total_considered``.

        Raises
        ------
        EmbeddingsError
            If embeddings collection is empty or operation fails.
        """
        if ids is not None and not ids:
            raise EmbeddingsError("No papers matched the filters")
        if self.collection.count() == 0:
            raise EmbeddingsError("No papers in collection")

        with _translate_errors("Failed to query papers within distance"):
            paper_ids, distances, embedding = self._query_embeddings(
                query,
                ids=ids,
                query_embedding=query_embedding,
            )
            if not paper_ids:
                raise EmbeddingsError("No results from collection query")

            matches = list(takewhile(lambda pair: pair[1] <= distance_threshold, zip(paper_ids, distances)))
            return {
                "query_embedding": embedding,
                "ids": [paper_id for paper_id, _ in matches],
                "distances": [float(distance) for _, distance in matches],
                "total_considered": len(paper_ids),
            }

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

        with _translate_errors("Failed to delete embeddings"):
            deleted = 0
            for batch in _iter_batches(ids, _IO_BATCH_SIZE):
                self.collection.delete(ids=list(batch))
                deleted += len(batch)
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
            If the export fails or the collection has not been compacted.
        """
        with _translate_errors("Failed to export embeddings"):
            exported_ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []

            for batch in _iter_batches(ids, _IO_BATCH_SIZE):
                results = self.collection.get(ids=list(batch), include=["documents", "embeddings"])
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
                            "Run 'python scripts/migrate_embeddings_compaction.py' before exporting."
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

        with _translate_errors("Failed to import embeddings"):
            if replace_ids:
                self.delete_embeddings_by_ids(replace_ids)

            imported = 0
            for batch in _iter_batches(list(zip(ids, documents, embeddings)), batch_size):
                self._store_embeddings(
                    ids=[item[0] for item in batch],
                    documents=[item[1] for item in batch],
                    embeddings=[item[2] for item in batch],
                )
                imported += len(batch)
            return imported
