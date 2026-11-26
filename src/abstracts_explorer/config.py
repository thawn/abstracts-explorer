"""
Configuration management for abstracts-explorer package.

This module loads configuration from environment variables and .env files.
Uses only standard library (no python-dotenv dependency required).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Uses a simple parser that handles basic .env file format without
    requiring external dependencies.

    Parameters
    ----------
    env_path : Path, optional
        Path to .env file. If None, looks for .env in current directory
        and parent directories up to the package root.

    Returns
    -------
    dict
        Dictionary of environment variables loaded from file.

    Examples
    --------
    >>> env_vars = load_env_file(Path(".env"))
    >>> print(env_vars.get("CHAT_MODEL"))
    """
    if env_path is None:
        # Look for .env file starting from current directory
        current = Path.cwd()
        for _ in range(5):  # Check up to 5 parent directories
            env_file = current / ".env"
            if env_file.exists():
                env_path = env_file
                break
            current = current.parent

    if env_path is None or not env_path.exists():
        return {}

    env_vars = {}
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]

                    env_vars[key] = value
    except Exception:
        # Silently ignore errors reading .env file
        pass

    return env_vars


class Config:
    """
    Configuration manager for abstracts-explorer package.

    Loads configuration from environment variables with fallback to defaults.
    Automatically loads from .env file if present.

    Attributes
    ----------
    data_dir : str
        Base directory for data files (databases, embeddings).
    read_only : bool
        Whether to run in read-only mode.  In this mode the paper database and
        the embedding store are opened without writing, schema creation is
        skipped, and every write operation raises ``ReadOnlyError``.  Multiple
        read-only instances may run concurrently against the same data.
    chat_model : str
        Name of the language model for chat/RAG.
    embedding_model : str
        Name of the embedding model.
    embedding_max_input_tokens : int
        Maximum number of tokens sent to the embedding model per request.
        Longer inputs are truncated before the API call. Set to 0 to disable
        proactive truncation.
    llm_backend_url : str
        URL for OpenAI-compatible API endpoint.
    llm_backend_auth_token : str
        Authentication token for LLM backend (if required).
    embedding_db : str
        ChromaDB configuration - can be either a URL (e.g., "http://chromadb:8000")
        or a file path (e.g., "chroma_db" or "/path/to/chroma_db").
    min_free_disk_mb : int
        Minimum free disk space in MiB required before writing embeddings
        (default: 4096). Local ChromaDB writes are refused when the
        filesystem has less space available. Set to 0 to disable the check.
    paper_db_path : str
        Path to SQLite paper database.
    collection_name : str
        Name of the ChromaDB collection.
    max_context_papers : int
        Default number of papers for RAG context.
    metadata_cache_ttl : int
        Time-to-live in seconds for the PostgreSQL metadata cache. ``0``
        disables the cache.
    rag_max_parallel_tools : int
        Maximum number of RAG tool calls executed concurrently (default: 8).
        Excess calls are queued by the agent's thread pool.
    rag_request_limit : int
        Maximum number of model requests per RAG query (default: 50).
    rag_query_timeout : int
        Wall-clock timeout in seconds for one RAG query (default: 120).
    chat_history_ttl : int
        Idle time in seconds after which a web chat session's history is
        expired (default: 86400). ``0`` disables expiry.
    llm_connect_timeout : int
        TCP/TLS connect timeout in seconds for LLM and embedding HTTP
        requests (default: 10).
    llm_request_timeout : int
        Timeout in seconds for one HTTP request to the LLM backend
        (default: 60). Clamped to ``rag_query_timeout`` so it never exceeds
        the overall RAG limit.
    chat_temperature : float
        Default temperature for chat generation.
    chat_max_tokens : int
        Default max tokens for chat responses.
    cluster_label_model : str
        Model used for short cluster-label generation during clustering.
        Defaults to ``alias-fast`` so a non-thinking instruction model is
        used for these one-shot label tasks (unlike ``chat_model``, which
        may be a thinking-capable agent model).
    cluster_label_max_workers : int
        Maximum number of cluster-label LLM requests submitted in parallel
        per labeling batch (default: 8). Set to 1 to label sequentially.
        Total request throughput is still bounded by
        ``requests_per_minute``.
    llm_max_attempts : int
        Total HTTP attempts for any LLM request (default: 6). Maps to the
        OpenAI SDK's ``max_retries = llm_max_attempts - 1``; the SDK owns
        retryable status classification, ``Retry-After`` handling, and
        backoff. Set to 1 to disable retries.
    llm_retry_base_delay : float
        Base delay in seconds for retrying cluster-label responses without
        usable content (default: 1.0). HTTP backoff is handled by the SDK.
    llm_retry_max_delay : float
        Maximum delay in seconds for empty-content cluster-label retries
        (default: 10.0).
    llm_rate_limit_delay : float
        Shared cooldown applied after HTTP 429 responses when the backend does
        not provide a usable ``Retry-After`` header (default: 60.0).
    llm_adaptive_rate_limit : bool
        Whether the shared rate limiter lowers its effective request rate after
        sustained ``429`` responses and recovers on success (default True).
    llm_adaptive_min_rpm : int
        Lower bound for the adaptively throttled request rate (default 6).
    enable_query_rewriting : bool
        Whether to enable query rewriting for better semantic search.
    query_similarity_threshold : float
        Similarity threshold for determining when to retrieve new papers (0.0-1.0).
    guard_enabled : bool
        Whether to classify chat input with a local Ollaya decision model before
        running the RAG agent (default: False).
    guard_backend_url : str
        Base URL of the Ollaya server (default: ``http://127.0.0.1:11435``).
    guard_api_key : str
        Optional bearer token for the Ollaya server.
    guard_model : str
        Ollaya model used for the guard decision (default: ``laya:multilingual``).
    guard_timeout : float
        Per-check HTTP timeout in seconds (default: 2.0).
    guard_topic_threshold : float
        Minimum probability for the abuse/off-topic choice to block a message
        (default: 0.8).
    guard_jailbreak_threshold : float
        Minimum probability for the jailbreak yes/no question to block a message
        (default: 0.8).
    guard_max_chars : int
        Maximum number of characters sent to Ollaya (default: 1500).
    guard_refusal_message : str
        Optional custom refusal text. Empty uses the built-in message.
    database_url : str
        SQLAlchemy database URL (supports SQLite, PostgreSQL, etc.).
        Automatically constructed from PAPER_DB config variable.
    log_level : str
        Logging level from environment (WARNING, INFO, DEBUG). Empty string if not set.
        Used by setup_logging() to set the default log level when verbosity flags are not used.
    proxy_x_for : int
        Number of trusted ``X-Forwarded-For`` proxy hops (ProxyFix).
        Set to 0 to disable; set to the number of reverse-proxy layers in front of the app.
    proxy_x_proto : int
        Number of trusted ``X-Forwarded-Proto`` proxy hops (ProxyFix).
    proxy_x_host : int
        Number of trusted ``X-Forwarded-Host`` proxy hops (ProxyFix).
    proxy_x_prefix : int
        Number of trusted ``X-Forwarded-Prefix`` proxy hops (ProxyFix).
    imprint_link : str
        Optional URL for an imprint/legal notice page. If set, a link is shown in the web UI footer.
        Empty string by default (no imprint link shown).
    requests_per_minute : int
        Maximum number of embedding API requests per minute (default: 60).
        Set to 0 to disable rate limiting.
    web_rate_limit_per_minute : int
        Maximum number of expensive web API requests per client per minute
        (chat, semantic search, cluster search, and topic evolution).
        Set to 0 to disable client rate limiting (default: 30).
    web_ui_threads : int
        Number of Waitress worker threads for the web UI (default: 6).
        Values below 1 are clamped to 1. Ignored when the Flask development
        server is used.
    openreview_email : str
        Email address for OpenReview login, used by the ICLR downloader.
    openreview_password : str
        Password for OpenReview login, used by the ICLR downloader.
    openreview_token : str
        Pre-existing OpenReview bearer token. When set, it is used instead of
        logging in with an email and password.

    Examples
    --------
    >>> config = Config()
    >>> print(config.chat_model)
    'diffbot-small-xl-2508'
    >>> config.llm_backend_url
    'http://localhost:1234'
    >>> # Using DATABASE_URL for PostgreSQL
    >>> config.database_url
    'postgresql://user:password@localhost/abstracts'
    """

    def __init__(self, env_path: Optional[Path] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        env_path : Path, optional
            Path to .env file. If None, searches for .env automatically.
        """
        # Load .env file if it exists
        env_vars = load_env_file(env_path)

        # Merge with actual environment variables (environment variables take precedence)
        self._env = {**env_vars, **os.environ}

        # Load all configuration values
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""
        # Data Directory (base directory for all data files)
        self.data_dir = self._get_env("DATA_DIR", default="data")

        # Read-only mode: refuse database/embedding writes and allow multiple
        # concurrent readers.
        self.read_only = self._get_env_bool("READ_ONLY", default=False)

        # Chat/Language Model Settings
        self.chat_model = self._get_env("CHAT_MODEL", default="diffbot-small-xl-2508")
        self.chat_temperature = self._get_env_float("CHAT_TEMPERATURE", default=0.7)
        self.chat_max_tokens = self._get_env_int("CHAT_MAX_TOKENS", default=4000)
        self.cluster_label_model = self._get_env("CLUSTER_LABEL_MODEL", default="alias-fast")

        # Embedding Model Settings
        self.embedding_model = self._get_env("EMBEDDING_MODEL", default="text-embedding-qwen3-embedding-4b")
        self.embedding_max_input_tokens = self._get_env_int("EMBEDDING_MAX_INPUT_TOKENS", default=8192)

        # LLM Backend Configuration
        self.llm_backend_url = self._get_env("LLM_BACKEND_URL", default="http://localhost:1234")
        self.llm_backend_auth_token = self._get_env("LLM_BACKEND_AUTH_TOKEN", default="")

        # Database Configuration
        # PAPER_DB can be either:
        # 1. A PostgreSQL URL (e.g., "postgresql://user:pass@host/db")
        # 2. A file path for SQLite (e.g., "abstracts.db" or "/path/to/abstracts.db")
        paper_db = self._get_env("PAPER_DB", default="abstracts.db")

        # Support Podman/Docker deployments where the database password is
        # injected as a separate secret (ABSTRACTS_DB_PASSWORD) rather than
        # embedded in PAPER_DB.  If PAPER_DB is still a plain file path (i.e.
        # no URL scheme) and all PostgreSQL component variables are present,
        # assemble the full connection URL automatically.
        #
        # Additionally, if POSTGRES_HOST is explicitly set in the process
        # environment (i.e., injected at runtime via a quadlet EnvironmentFile
        # or similar mechanism, rather than read from a baked-in .env file),
        # it takes precedence over any PAPER_DB value from a .env file.  This
        # allows Podman quadlet deployments to override the default
        # PAPER_DB embedded in the container image via individual POSTGRES_*
        # variables without having to re-specify the full connection URL.
        postgres_host_in_env = "POSTGRES_HOST" in os.environ
        if "://" not in paper_db or postgres_host_in_env:
            pg_user = self._get_env("POSTGRES_USER", default="")
            pg_password = self._get_env("ABSTRACTS_DB_PASSWORD", default="")
            pg_host = self._get_env("POSTGRES_HOST", default="")
            pg_port = self._get_env("POSTGRES_PORT", default="5432")
            pg_db = self._get_env("POSTGRES_DB", default="")
            if pg_user and pg_password and pg_host and pg_db:
                paper_db = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

        if paper_db.startswith("postgresql://") or paper_db.startswith("sqlite://"):
            # Full database URL provided
            self.database_url = paper_db
        elif "://" in paper_db:
            # Other database URL (mysql, etc.)
            self.database_url = paper_db
        else:
            # File path - treat as SQLite
            paper_db_path = self._resolve_path(paper_db)
            self.database_url = f"sqlite:///{paper_db_path}"

        # Embedding database configuration
        # EMBEDDING_DB can be either a URL (e.g., "http://chromadb:8000")
        # or a file path (e.g., "chroma_db" or "/path/to/chroma_db")
        embedding_db = self._get_env("EMBEDDING_DB", default="chroma_db")

        if embedding_db.startswith("http://") or embedding_db.startswith("https://"):
            # URL provided - use as-is
            self.embedding_db = embedding_db
        else:
            # File path - resolve relative to data_dir
            self.embedding_db = self._resolve_path(embedding_db)

        # Minimum free disk space required before writing embeddings (MiB).
        # Writing a native vector index while the disk is full can corrupt it.
        self.min_free_disk_mb = max(0, self._get_env_int("MIN_FREE_DISK_MB", default=4096))

        # Collection Settings
        self.collection_name = self._get_env("COLLECTION_NAME", default="papers")

        # RAG Settings
        self.max_context_papers = self._get_env_int("MAX_CONTEXT_PAPERS", default=5)
        self.metadata_cache_ttl = self._get_env_int("METADATA_CACHE_TTL", default=86400)
        self.rag_max_parallel_tools = max(1, self._get_env_int("RAG_MAX_PARALLEL_TOOLS", default=8))
        self.rag_request_limit = max(1, self._get_env_int("RAG_REQUEST_LIMIT", default=50))
        self.rag_query_timeout = max(1, self._get_env_int("RAG_QUERY_TIMEOUT", default=120))
        self.chat_history_ttl = max(0, self._get_env_int("CHAT_HISTORY_TTL", default=86400))

        # LLM HTTP Timeout Settings
        # Per-request timeout is clamped to the overall RAG query timeout so a
        # single HTTP request can never outlive the RAG query ceiling.
        self.llm_connect_timeout = max(
            1,
            self._get_env_int("LLM_CONNECT_TIMEOUT", default=10),
        )
        self.llm_request_timeout = min(
            max(1, self._get_env_int("LLM_REQUEST_TIMEOUT", default=60)),
            self.rag_query_timeout,
        )

        # LLM concurrency and retry settings
        self.cluster_label_max_workers = max(1, self._get_env_int("CLUSTER_LABEL_MAX_WORKERS", default=8))
        self.llm_max_attempts = max(1, self._get_env_int("LLM_MAX_ATTEMPTS", default=6))
        self.llm_retry_base_delay = max(0.0, self._get_env_float("LLM_RETRY_BASE_DELAY", default=1.0))
        self.llm_retry_max_delay = max(0.0, self._get_env_float("LLM_RETRY_MAX_DELAY", default=10.0))
        self.llm_rate_limit_delay = max(0.0, self._get_env_float("LLM_RATE_LIMIT_DELAY", default=60.0))
        self.llm_adaptive_rate_limit = self._get_env_bool("ADAPTIVE_RATE_LIMIT_ENABLED", default=True)
        self.llm_adaptive_min_rpm = max(1, self._get_env_int("ADAPTIVE_MIN_RPM", default=6))

        # Query Rewriting Settings
        self.enable_query_rewriting = self._get_env_bool("ENABLE_QUERY_REWRITING", default=True)
        self.query_similarity_threshold = self._get_env_float("QUERY_SIMILARITY_THRESHOLD", default=0.7)

        # Chat Input Guard Settings (Ollaya decision model)
        self.guard_enabled = self._get_env_bool("GUARD_ENABLED", default=False)
        self.guard_backend_url = self._get_env("GUARD_BACKEND_URL", default="http://127.0.0.1:11435")
        self.guard_api_key = self._get_env("GUARD_API_KEY", default="")
        self.guard_model = self._get_env("GUARD_MODEL", default="laya:multilingual")
        self.guard_timeout = max(0.1, self._get_env_float("GUARD_TIMEOUT", default=2.0))
        self.guard_topic_threshold = min(1.0, max(0.0, self._get_env_float("GUARD_TOPIC_THRESHOLD", default=0.8)))
        self.guard_jailbreak_threshold = min(
            1.0, max(0.0, self._get_env_float("GUARD_JAILBREAK_THRESHOLD", default=0.8))
        )
        self.guard_max_chars = max(1, self._get_env_int("GUARD_MAX_CHARS", default=1500))
        self.guard_refusal_message = self._get_env("GUARD_REFUSAL_MESSAGE", default="")

        # Web UI Default Filters
        self.default_conference = self._get_env("DEFAULT_CONFERENCE", default="ML4PS")
        self.default_year = self._get_env_int("DEFAULT_YEAR", default=0)

        # Logging Configuration
        self.log_level = self._get_env("LOG_LEVEL", default="").upper()

        # Reverse-proxy hop counts for werkzeug ProxyFix.
        # Default to 0 (trust no forwarded headers). Deployments behind a
        # reverse proxy must set the number of trusted hops explicitly.
        self.proxy_x_for = self._get_env_int("PROXY_X_FOR", default=0)
        self.proxy_x_proto = self._get_env_int("PROXY_X_PROTO", default=0)
        self.proxy_x_host = self._get_env_int("PROXY_X_HOST", default=0)
        self.proxy_x_prefix = self._get_env_int("PROXY_X_PREFIX", default=0)

        # Web UI Footer Configuration
        # Optional imprint/legal notice URL shown in the footer (empty = not shown)
        self.imprint_link = self._get_env("IMPRINT_LINK", default="")

        # Embedding API Rate Limiting
        # Maximum number of API requests per minute (0 = no limit)
        self.requests_per_minute = self._get_env_int("REQUESTS_PER_MINUTE", default=60)

        # Web API client rate limiting for expensive endpoints (0 = disabled)
        self.web_rate_limit_per_minute = max(0, self._get_env_int("WEB_RATE_LIMIT_PER_MINUTE", default=30))

        # Waitress worker threads for the web UI
        self.web_ui_threads = max(1, self._get_env_int("WEB_UI_THREADS", default=6))

        # Registry Settings
        # GitHub Personal Access Token for registry operations
        self.github_token = self._get_env("GITHUB_TOKEN", default="")
        # OCI repository for data artifacts (e.g., 'ghcr.io/thawn/abstracts-data')
        self.registry_repository = self._get_env("REGISTRY_REPOSITORY", default="")

        # OpenReview authentication (ICLR downloader)
        self.openreview_email = self._get_env("OPENREVIEW_EMAIL", default="")
        self.openreview_password = self._get_env("OPENREVIEW_PASSWORD", default="")
        self.openreview_token = self._get_env("OPENREVIEW_TOKEN", default="")

    def _get_env(self, key: str, default: str = "") -> str:
        """
        Get string environment variable.

        Parameters
        ----------
        key : str
            Environment variable name.
        default : str
            Default value if not set.

        Returns
        -------
        str
            Environment variable value or default.
        """
        return self._env.get(key, default)

    def _get_env_int(self, key: str, default: int = 0) -> int:
        """
        Get integer environment variable.

        Parameters
        ----------
        key : str
            Environment variable name.
        default : int
            Default value if not set or invalid.

        Returns
        -------
        int
            Environment variable value as integer or default.
        """
        value = self._env.get(key, "")
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _get_env_float(self, key: str, default: float = 0.0) -> float:
        """
        Get float environment variable.

        Parameters
        ----------
        key : str
            Environment variable name.
        default : float
            Default value if not set or invalid.

        Returns
        -------
        float
            Environment variable value as float or default.
        """
        value = self._env.get(key, "")
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """
        Get boolean environment variable.

        Parameters
        ----------
        key : str
            Environment variable name.
        default : bool
            Default value if not set or invalid.

        Returns
        -------
        bool
            Environment variable value as boolean or default.
        """
        value = self._env.get(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        return default

    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the data directory.

        If the path is absolute, returns it unchanged.
        Otherwise, resolves it relative to data_dir.

        Parameters
        ----------
        path : str
            Path to resolve.

        Returns
        -------
        str
            Resolved path (relative to data_dir if not absolute).
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            # Return absolute path as string (expanduser to handle ~)
            return str(path_obj.expanduser().absolute())

        # Resolve relative to data_dir and make absolute
        return str((Path(self.data_dir) / path).absolute())

    @property
    def masked_database_url(self) -> str:
        """Return the database URL with any password masked for display."""
        from abstracts_explorer.db.session import mask_database_url

        return mask_database_url(self.database_url)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary of all configuration values.

        Examples
        --------
        >>> config = Config()
        >>> config_dict = config.to_dict()
        >>> print(config_dict["chat_model"])
        """
        return {
            "data_dir": self.data_dir,
            "read_only": self.read_only,
            "chat_model": self.chat_model,
            "chat_temperature": self.chat_temperature,
            "chat_max_tokens": self.chat_max_tokens,
            "embedding_model": self.embedding_model,
            "embedding_max_input_tokens": self.embedding_max_input_tokens,
            "llm_backend_url": self.llm_backend_url,
            "llm_backend_auth_token": "***" if self.llm_backend_auth_token else "",
            "embedding_db": self.embedding_db,
            "min_free_disk_mb": self.min_free_disk_mb,
            "database_url": self._mask_database_url(self.database_url),
            "collection_name": self.collection_name,
            "max_context_papers": self.max_context_papers,
            "metadata_cache_ttl": self.metadata_cache_ttl,
            "rag_max_parallel_tools": self.rag_max_parallel_tools,
            "rag_request_limit": self.rag_request_limit,
            "rag_query_timeout": self.rag_query_timeout,
            "chat_history_ttl": self.chat_history_ttl,
            "llm_connect_timeout": self.llm_connect_timeout,
            "llm_request_timeout": self.llm_request_timeout,
            "llm_max_attempts": self.llm_max_attempts,
            "llm_retry_base_delay": self.llm_retry_base_delay,
            "llm_retry_max_delay": self.llm_retry_max_delay,
            "llm_rate_limit_delay": self.llm_rate_limit_delay,
            "llm_adaptive_rate_limit": self.llm_adaptive_rate_limit,
            "llm_adaptive_min_rpm": self.llm_adaptive_min_rpm,
            "web_rate_limit_per_minute": self.web_rate_limit_per_minute,
            "web_ui_threads": self.web_ui_threads,
            "guard_enabled": self.guard_enabled,
            "guard_backend_url": self.guard_backend_url,
            "guard_api_key": "***" if self.guard_api_key else "",
            "guard_model": self.guard_model,
            "guard_timeout": self.guard_timeout,
            "guard_topic_threshold": self.guard_topic_threshold,
            "guard_jailbreak_threshold": self.guard_jailbreak_threshold,
            "guard_max_chars": self.guard_max_chars,
            "guard_refusal_message": self.guard_refusal_message,
            "openreview_email": self.openreview_email,
            "openreview_password": "***" if self.openreview_password else "",
            "openreview_token": "***" if self.openreview_token else "",
        }

    def _mask_database_url(self, url: str) -> str:
        """
        Mask password in database URL for display.

        Parameters
        ----------
        url : str
            Database URL that may contain password.

        Returns
        -------
        str
            URL with password masked.
        """
        if not url or "://" not in url:
            return url

        # For URLs with password (e.g., postgresql://user:password@host/db)
        # Mask the password part
        if "@" in url and ":" in url.split("://")[1].split("@")[0]:
            parts = url.split("://")
            protocol = parts[0]
            rest = parts[1]
            # Split at @ to separate credentials from host
            creds_and_host = rest.split("@")
            if len(creds_and_host) == 2:
                creds = creds_and_host[0]
                host = creds_and_host[1]
                # Mask password in credentials
                if ":" in creds:
                    user = creds.split(":")[0]
                    return f"{protocol}://{user}:***@{host}"
        return url

    def __repr__(self) -> str:
        """String representation of configuration."""
        items = []
        for key, value in self.to_dict().items():
            items.append(f"{key}={value}")
        return f"Config({', '.join(items)})"


# Global configuration instance
_config: Optional[Config] = None


def get_config(reload: bool = False, env_path: Optional[Path] = None) -> Config:
    """
    Get global configuration instance.

    Parameters
    ----------
    reload : bool, optional
        Force reload configuration from environment, by default False
    env_path : Path, optional
        Path to .env file. If provided, loads configuration from this file.
        Useful for testing to ensure consistent configuration.

    Returns
    -------
    Config
        Global configuration instance.

    Examples
    --------
    >>> config = get_config()
    >>> print(config.chat_model)
    >>> # In tests, use .env.tests for consistent values
    >>> config = get_config(reload=True, env_path=Path(".env.tests"))
    """
    global _config
    if _config is None or reload:
        _config = Config(env_path=env_path)
    return _config
