"""
Configuration management for neurips-abstracts package.

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
    Configuration manager for neurips-abstracts package.

    Loads configuration from environment variables with fallback to defaults.
    Automatically loads from .env file if present.

    Attributes
    ----------
    data_dir : str
        Base directory for data files (databases, embeddings).
    chat_model : str
        Name of the language model for chat/RAG.
    embedding_model : str
        Name of the embedding model.
    llm_backend_url : str
        URL for LM Studio or compatible API.
    llm_backend_auth_token : str
        Authentication token for LLM backend (if required).
    embedding_db_path : str
        Path to ChromaDB vector database.
    paper_db_path : str
        Path to SQLite paper database.
    collection_name : str
        Name of the ChromaDB collection.
    max_context_papers : int
        Default number of papers for RAG context.
    chat_temperature : float
        Default temperature for chat generation.
    chat_max_tokens : int
        Default max tokens for chat responses.
    enable_query_rewriting : bool
        Whether to enable query rewriting for better semantic search.
    query_similarity_threshold : float
        Similarity threshold for determining when to retrieve new papers (0.0-1.0).

    Examples
    --------
    >>> config = Config()
    >>> print(config.chat_model)
    'diffbot-small-xl-2508'
    >>> config.llm_backend_url
    'http://localhost:1234'
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

        # Merge with actual environment variables (env vars take precedence)
        self._env = {**env_vars, **os.environ}

        # Load all configuration values
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""
        # Data Directory (base directory for all data files)
        self.data_dir = self._get_env("DATA_DIR", default="data")

        # Chat/Language Model Settings
        self.chat_model = self._get_env("CHAT_MODEL", default="diffbot-small-xl-2508")
        self.chat_temperature = self._get_env_float("CHAT_TEMPERATURE", default=0.7)
        self.chat_max_tokens = self._get_env_int("CHAT_MAX_TOKENS", default=1000)

        # Embedding Model Settings
        self.embedding_model = self._get_env("EMBEDDING_MODEL", default="text-embedding-qwen3-embedding-4b")

        # LLM Backend Configuration
        self.llm_backend_url = self._get_env("LLM_BACKEND_URL", default="http://localhost:1234")
        self.llm_backend_auth_token = self._get_env("LLM_BACKEND_AUTH_TOKEN", default="")

        # Database Paths (resolved relative to data_dir if not absolute)
        self.embedding_db_path = self._resolve_path(self._get_env("EMBEDDING_DB_PATH", default="chroma_db"))
        self.paper_db_path = self._resolve_path(self._get_env("PAPER_DB_PATH", default="neurips_2025.db"))

        # Collection Settings
        self.collection_name = self._get_env("COLLECTION_NAME", default="neurips_papers")

        # RAG Settings
        self.max_context_papers = self._get_env_int("MAX_CONTEXT_PAPERS", default=5)

        # Query Rewriting Settings
        self.enable_query_rewriting = self._get_env_bool("ENABLE_QUERY_REWRITING", default=True)
        self.query_similarity_threshold = self._get_env_float("QUERY_SIMILARITY_THRESHOLD", default=0.7)

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
            "chat_model": self.chat_model,
            "chat_temperature": self.chat_temperature,
            "chat_max_tokens": self.chat_max_tokens,
            "embedding_model": self.embedding_model,
            "llm_backend_url": self.llm_backend_url,
            "llm_backend_auth_token": "***" if self.llm_backend_auth_token else "",
            "embedding_db_path": self.embedding_db_path,
            "paper_db_path": self.paper_db_path,
            "collection_name": self.collection_name,
            "max_context_papers": self.max_context_papers,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        items = []
        for key, value in self.to_dict().items():
            items.append(f"{key}={value}")
        return f"Config({', '.join(items)})"


# Global configuration instance
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Parameters
    ----------
    reload : bool, optional
        Force reload configuration from environment, by default False

    Returns
    -------
    Config
        Global configuration instance.

    Examples
    --------
    >>> config = get_config()
    >>> print(config.chat_model)
    """
    global _config
    if _config is None or reload:
        _config = Config()
    return _config
