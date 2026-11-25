"""
Embeddings manager facade.

:class:`EmbeddingsManager` composes the mixins that implement ChromaDB
storage, client/model setup, generation, queries, ingestion, and artifacts.
"""

import logging
import threading
from typing import Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.embeddings.artifacts import ArtifactsMixin
from abstracts_explorer.embeddings.chroma import ChromaMixin
from abstracts_explorer.embeddings.generation import GenerationMixin
from abstracts_explorer.embeddings.ingest import IngestMixin
from abstracts_explorer.embeddings.models import ModelMixin
from abstracts_explorer.embeddings.query import QueryMixin

logger = logging.getLogger(__name__)


class EmbeddingsManager(GenerationMixin, QueryMixin, IngestMixin, ArtifactsMixin, ModelMixin, ChromaMixin):
    """
    Manager for generating and storing text embeddings.

    This class handles:

    - Connecting to an OpenAI-compatible API for embedding generation
    - Creating and managing a ChromaDB collection
    - Embedding paper abstracts and storing their content hashes
    - Similarity search operations

    Parameters
    ----------
    backend_url : str, optional
        URL of the OpenAI-compatible API endpoint, by default "http://localhost:1234"
    auth_token : str, optional
        Authentication token for the backend, if required.
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
    >>> results = em.search_similar(QueryEmbedding("machine learning", em), limit=5)
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

        Parameters are optional and will use values from environment/config if
        not provided.

        Parameters
        ----------
        backend_url : str, optional
            URL of the OpenAI-compatible API endpoint. If None, uses config value.
        auth_token : str, optional
            Authentication token for the backend. If None, uses config value.
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
        self._client: Optional[object] = None
        self._collection: Optional[object] = None
        self._chroma_lock_path: Optional[str] = None
        self._init_lock = threading.RLock()
        self.min_free_disk_mb = config.min_free_disk_mb

        self._openai_client = None

        self.requests_per_minute = (
            requests_per_minute if requests_per_minute is not None else config.requests_per_minute
        )
        self.request_timeout = config.llm_request_timeout
        self.connect_timeout = config.llm_connect_timeout
        self.llm_max_attempts = config.llm_max_attempts
        self.llm_rate_limit_delay = config.llm_rate_limit_delay
        self.llm_adaptive_rate_limit = config.llm_adaptive_rate_limit
        self.llm_adaptive_min_rpm = config.llm_adaptive_min_rpm

    def __enter__(self) -> "EmbeddingsManager":
        """Context manager entry; connects to ChromaDB."""
        _ = self.client
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
