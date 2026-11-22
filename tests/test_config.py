"""
Tests for configuration module.

This module tests the configuration loading and management functionality.
"""

import os
from pathlib import Path

from abstracts_explorer.config import Config, get_config, load_env_file
from abstracts_explorer import config as config_module

from tests.conftest import get_base_env_test_path


def clean_config_env_vars():
    """Clean config-related environment variables.

    Uses an explicit list of supported environment variable names. This ensures that
    environment-only keys such as ``PAPER_DB`` are also removed.
    """
    env_vars = {
        "DATA_DIR",
        "CHAT_MODEL",
        "CHAT_TEMPERATURE",
        "CHAT_MAX_TOKENS",
        "CLUSTER_LABEL_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_MAX_INPUT_TOKENS",
        "LLM_BACKEND_URL",
        "LLM_BACKEND_AUTH_TOKEN",
        "PAPER_DB",
        "POSTGRES_USER",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "ABSTRACTS_DB_PASSWORD",
        "EMBEDDING_DB",
        "COLLECTION_NAME",
        "MAX_CONTEXT_PAPERS",
        "METADATA_CACHE_TTL",
        "ENABLE_QUERY_REWRITING",
        "QUERY_SIMILARITY_THRESHOLD",
        "DEFAULT_CONFERENCE",
        "DEFAULT_YEAR",
        "LOG_LEVEL",
        "PROXY_X_FOR",
        "PROXY_X_PROTO",
        "PROXY_X_HOST",
        "PROXY_X_PREFIX",
        "IMPRINT_LINK",
        "REQUESTS_PER_MINUTE",
        "WEB_RATE_LIMIT_PER_MINUTE",
        "WEB_UI_THREADS",
        "RAG_MAX_PARALLEL_TOOLS",
        "RAG_REQUEST_LIMIT",
        "RAG_QUERY_TIMEOUT",
        "CHAT_HISTORY_TTL",
        "LLM_CONNECT_TIMEOUT",
        "LLM_REQUEST_TIMEOUT",
        "GITHUB_TOKEN",
        "REGISTRY_REPOSITORY",
        "OPENREVIEW_EMAIL",
        "OPENREVIEW_PASSWORD",
        "OPENREVIEW_TOKEN",
        "GUARD_ENABLED",
        "GUARD_BACKEND_URL",
        "GUARD_API_KEY",
        "GUARD_MODEL",
        "GUARD_TIMEOUT",
        "GUARD_TOPIC_THRESHOLD",
        "GUARD_JAILBREAK_THRESHOLD",
        "GUARD_MAX_CHARS",
        "GUARD_REFUSAL_MESSAGE",
    }
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]


class TestLoadEnvFile:
    """Test .env file loading."""

    def test_load_env_file_basic(self, tmp_path):
        """Test loading a basic .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
# Comment line
CHAT_MODEL=test-model
EMBEDDING_MODEL=test-embedding
LLM_BACKEND_URL=http://test:8080

# Another comment
MAX_CONTEXT_PAPERS=10
""")

        env_vars = load_env_file(env_file)

        assert env_vars["CHAT_MODEL"] == "test-model"
        assert env_vars["EMBEDDING_MODEL"] == "test-embedding"
        assert env_vars["LLM_BACKEND_URL"] == "http://test:8080"
        assert env_vars["MAX_CONTEXT_PAPERS"] == "10"

    def test_load_env_file_with_quotes(self, tmp_path):
        """Test loading .env file with quoted values."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
CHAT_MODEL="quoted-model"
EMBEDDING_MODEL='single-quoted'
LLM_BACKEND_URL=http://localhost:1234
""")

        env_vars = load_env_file(env_file)

        assert env_vars["CHAT_MODEL"] == "quoted-model"
        assert env_vars["EMBEDDING_MODEL"] == "single-quoted"
        assert env_vars["LLM_BACKEND_URL"] == "http://localhost:1234"

    def test_load_env_file_empty_lines(self, tmp_path):
        """Test loading .env file with empty lines."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
CHAT_MODEL=model1

EMBEDDING_MODEL=model2


LLM_BACKEND_URL=http://test
""")

        env_vars = load_env_file(env_file)

        assert len(env_vars) == 3
        assert env_vars["CHAT_MODEL"] == "model1"

    def test_load_env_file_not_found(self, tmp_path):
        """Test loading non-existent .env file."""
        env_vars = load_env_file(tmp_path / "nonexistent.env")
        assert env_vars == {}

    def test_load_env_file_none(self):
        """Test loading .env file with None path."""
        env_vars = load_env_file(None)
        # Should return empty dict if no .env found in search path
        assert isinstance(env_vars, dict)


class TestEnvTestLocalOverride:
    """Test .env.tests.local selection in the test setup."""

    def setup_method(self):
        """Clear environment variables before each test to ensure isolation."""
        clean_config_env_vars()

    def teardown_method(self):
        """Clean up environment variables after each test."""
        clean_config_env_vars()

    def _patch_base_path(self, monkeypatch, tmp_path, base_content, local_content=None):
        """Create base/local env files and point the helper at them."""
        from tests import conftest as conftest_module

        base_path = tmp_path / ".env.tests"
        base_path.write_text(base_content)
        if local_content is not None:
            (tmp_path / ".env.tests.local").write_text(local_content)
        monkeypatch.setattr(conftest_module, "get_base_env_test_path", lambda: base_path)
        return conftest_module

    def test_local_file_is_used_instead_of_base(self, monkeypatch, tmp_path):
        """When .env.tests.local exists, it replaces the tracked base file."""
        conftest_module = self._patch_base_path(
            monkeypatch,
            tmp_path,
            "CHAT_MODEL=base-model\nEMBEDDING_MODEL=base-embedding\n",
            "CHAT_MODEL=local-model\nEMBEDDING_MODEL=local-embedding\n",
        )

        env_path = conftest_module.get_env_test_path()
        config = Config(env_path=env_path)

        assert env_path == tmp_path / ".env.tests.local"
        assert config.chat_model == "local-model"
        assert config.embedding_model == "local-embedding"

    def test_missing_local_file_returns_base_path(self, monkeypatch, tmp_path):
        """Without a local file, the tracked base path is returned unchanged."""
        conftest_module = self._patch_base_path(monkeypatch, tmp_path, "CHAT_MODEL=base-model\n")

        assert conftest_module.get_env_test_path() == tmp_path / ".env.tests"

    def test_process_env_overrides_local_file(self, monkeypatch, tmp_path):
        """Real environment variables still beat the local env file."""
        conftest_module = self._patch_base_path(
            monkeypatch,
            tmp_path,
            "CHAT_MODEL=base-model\n",
            "CHAT_MODEL=local-model\n",
        )
        monkeypatch.setenv("CHAT_MODEL", "env-model")

        config = Config(env_path=conftest_module.get_env_test_path())

        assert config.chat_model == "env-model"


class TestConfig:
    """Test Config class."""

    def setup_method(self):
        """Clear environment variables before each test to ensure isolation."""
        # List of all config-related environment variables
        clean_config_env_vars()

    def teardown_method(self):
        """Clean up environment variables after each test."""
        # List of all config-related environment variables
        clean_config_env_vars()

    def test_config_defaults(self, monkeypatch):
        """Test default configuration values."""
        # Clear all environment variables that might override defaults

        # Use .env.tests which has the default values, ignoring local overrides
        config = Config(env_path=get_base_env_test_path())

        assert config.data_dir == "data"
        assert config.chat_model == "deepseek-v4.1-flash-at-hzdr"
        assert config.embedding_model == "alias-qwen3-8b-embeddings"
        assert config.llm_backend_url == "https://api.helmholtz-blablador.fz-juelich.de"
        assert config.embedding_db == str((Path("data") / "chroma_db").absolute())
        # Test database_url is constructed from PAPER_DB
        assert config.database_url == f"sqlite:///{(Path('data') / 'abstracts.db').absolute()}"
        assert config.collection_name == "papers"
        assert config.max_context_papers == 5
        assert config.metadata_cache_ttl == 86400
        assert config.chat_temperature == 0.7
        assert config.chat_max_tokens == 4000
        assert config.cluster_label_model == "alias-fast"
        assert config.embedding_max_input_tokens == 8192
        assert config.rag_max_parallel_tools == 8
        assert config.rag_request_limit == 50
        assert config.rag_query_timeout == 120
        assert config.chat_history_ttl == 86400
        assert config.llm_connect_timeout == 10
        assert config.llm_request_timeout == 60
        assert config.proxy_x_for == 0
        assert config.proxy_x_proto == 0
        assert config.proxy_x_host == 0
        assert config.proxy_x_prefix == 0
        assert config.web_rate_limit_per_minute == 30
        assert config.web_ui_threads == 6
        assert config.guard_enabled is False
        assert config.guard_backend_url == "http://127.0.0.1:11435"
        assert config.guard_api_key == ""
        assert config.guard_model == "laya:multilingual"
        assert config.guard_timeout == 2.0
        assert config.guard_topic_threshold == 0.8
        assert config.guard_jailbreak_threshold == 0.8
        assert config.guard_max_chars == 1500
        assert config.guard_refusal_message == ""

    def test_web_ui_threads_from_env(self, tmp_path):
        """WEB_UI_THREADS overrides the default Waitress thread count."""
        env_file = tmp_path / ".env"
        env_file.write_text("WEB_UI_THREADS=12\n")

        config = Config(env_path=env_file)
        assert config.web_ui_threads == 12

    def test_web_ui_threads_are_clamped(self, tmp_path):
        """Invalid or non-positive WEB_UI_THREADS values fall back safely."""
        env_file = tmp_path / ".env"
        env_file.write_text("WEB_UI_THREADS=0\n")

        config = Config(env_path=env_file)
        assert config.web_ui_threads == 1

        env_file.write_text("WEB_UI_THREADS=-4\n")
        config = Config(env_path=env_file)
        assert config.web_ui_threads == 1

        env_file.write_text("WEB_UI_THREADS=invalid\n")
        config = Config(env_path=env_file)
        assert config.web_ui_threads == 6

    def test_embedding_max_input_tokens_from_env(self, tmp_path):
        """Test parsing EMBEDDING_MAX_INPUT_TOKENS."""
        env_file = tmp_path / ".env"
        env_file.write_text("EMBEDDING_MAX_INPUT_TOKENS=2048\n")

        config = Config(env_path=env_file)
        assert config.embedding_max_input_tokens == 2048

    def test_embedding_max_input_tokens_zero(self, tmp_path):
        """A zero budget disables proactive truncation."""
        env_file = tmp_path / ".env"
        env_file.write_text("EMBEDDING_MAX_INPUT_TOKENS=0\n")

        config = Config(env_path=env_file)
        assert config.embedding_max_input_tokens == 0

    def test_rag_settings_from_env(self, tmp_path):
        """RAG concurrency and budget settings are configurable."""
        env_file = tmp_path / ".env"
        env_file.write_text("RAG_MAX_PARALLEL_TOOLS=3\nRAG_REQUEST_LIMIT=20\nRAG_QUERY_TIMEOUT=45\n")

        config = Config(env_path=env_file)
        assert config.rag_max_parallel_tools == 3
        assert config.rag_request_limit == 20
        assert config.rag_query_timeout == 45

    def test_llm_timeouts_from_env(self, tmp_path):
        """LLM connect and request timeouts are configurable."""
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_CONNECT_TIMEOUT=5\nLLM_REQUEST_TIMEOUT=45\n")

        config = Config(env_path=env_file)
        assert config.llm_connect_timeout == 5
        assert config.llm_request_timeout == 45

    def test_llm_request_timeout_clamped_to_query_timeout(self, tmp_path):
        """LLM request timeout cannot exceed the overall RAG query timeout."""
        env_file = tmp_path / ".env"
        env_file.write_text("RAG_QUERY_TIMEOUT=120\nLLM_REQUEST_TIMEOUT=240\n")

        config = Config(env_path=env_file)
        assert config.rag_query_timeout == 120
        assert config.llm_request_timeout == 120

    def test_rag_settings_are_clamped(self, tmp_path):
        """Non-positive RAG settings fall back to safe minimums."""
        env_file = tmp_path / ".env"
        env_file.write_text("RAG_MAX_PARALLEL_TOOLS=0\nRAG_REQUEST_LIMIT=-1\nRAG_QUERY_TIMEOUT=0\n")

        config = Config(env_path=env_file)
        assert config.rag_max_parallel_tools == 1
        assert config.rag_request_limit == 1
        assert config.rag_query_timeout == 1

    def test_metadata_cache_ttl_from_env(self, tmp_path):
        """METADATA_CACHE_TTL overrides the default."""
        env_file = tmp_path / ".env"
        env_file.write_text("METADATA_CACHE_TTL=120\n")

        config = Config(env_path=env_file)
        assert config.metadata_cache_ttl == 120

    def test_chat_history_ttl_from_env(self, tmp_path):
        """CHAT_HISTORY_TTL overrides the default and accepts 0 to disable expiry."""
        env_file = tmp_path / ".env"
        env_file.write_text("CHAT_HISTORY_TTL=3600\n")

        config = Config(env_path=env_file)
        assert config.chat_history_ttl == 3600

        env_file.write_text("CHAT_HISTORY_TTL=0\n")
        config = Config(env_path=env_file)
        assert config.chat_history_ttl == 0

    def test_config_from_env_file(self, tmp_path):
        """Test loading configuration from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
CHAT_MODEL=custom-chat-model
EMBEDDING_MODEL=custom-embedding
LLM_BACKEND_URL=http://custom:9999
LLM_BACKEND_AUTH_TOKEN=secret-token
EMBEDDING_DB=custom_chroma
PAPER_DB=custom.db
COLLECTION_NAME=custom_collection
MAX_CONTEXT_PAPERS=15
CHAT_TEMPERATURE=0.9
CHAT_MAX_TOKENS=4000
CLUSTER_LABEL_MODEL=custom-label-model
""")

        config = Config(env_path=env_file)

        assert config.chat_model == "custom-chat-model"
        assert config.embedding_model == "custom-embedding"
        assert config.llm_backend_url == "http://custom:9999"
        assert config.llm_backend_auth_token == "secret-token"
        # Paths are resolved relative to data_dir
        assert config.embedding_db == str((Path("data") / "custom_chroma").absolute())
        # Test database_url is constructed from PAPER_DB
        assert config.database_url == f"sqlite:///{(Path('data') / 'custom.db').absolute()}"
        assert config.collection_name == "custom_collection"
        assert config.max_context_papers == 15
        assert config.chat_temperature == 0.9
        assert config.chat_max_tokens == 4000
        assert config.cluster_label_model == "custom-label-model"

    def test_config_env_vars_override_file(self, tmp_path, monkeypatch):
        """Test that environment variables override .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("CHAT_MODEL=file-model")

        monkeypatch.setenv("CHAT_MODEL", "env-model")

        config = Config(env_path=env_file)
        assert config.chat_model == "env-model"

    def test_config_type_conversion(self, tmp_path):
        """Test configuration type conversion for int and float."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
MAX_CONTEXT_PAPERS=20
CHAT_TEMPERATURE=0.5
CHAT_MAX_TOKENS=500
""")

        config = Config(env_path=env_file)

        assert isinstance(config.max_context_papers, int)
        assert config.max_context_papers == 20

        assert isinstance(config.chat_temperature, float)
        assert config.chat_temperature == 0.5

        assert isinstance(config.chat_max_tokens, int)
        assert config.chat_max_tokens == 500

    def test_config_invalid_int_falls_back(self, tmp_path):
        """Test that invalid int values fall back to defaults."""
        env_file = tmp_path / ".env"
        env_file.write_text("MAX_CONTEXT_PAPERS=invalid")

        config = Config(env_path=env_file)
        assert config.max_context_papers == 5  # Default

    def test_config_invalid_float_falls_back(self, tmp_path):
        """Test that invalid float values fall back to defaults."""
        env_file = tmp_path / ".env"
        env_file.write_text("CHAT_TEMPERATURE=invalid")

        config = Config(env_path=env_file)
        assert config.chat_temperature == 0.7  # Default

    def test_guard_settings_from_env(self, tmp_path, monkeypatch):
        """Test custom guard settings load from the environment."""
        for key in (
            "GUARD_ENABLED",
            "GUARD_BACKEND_URL",
            "GUARD_API_KEY",
            "GUARD_MODEL",
            "GUARD_TIMEOUT",
            "GUARD_TOPIC_THRESHOLD",
            "GUARD_JAILBREAK_THRESHOLD",
            "GUARD_MAX_CHARS",
            "GUARD_REFUSAL_MESSAGE",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("GUARD_ENABLED", "true")
        monkeypatch.setenv("GUARD_BACKEND_URL", "http://guard:9000")
        monkeypatch.setenv("GUARD_API_KEY", "secret")
        monkeypatch.setenv("GUARD_MODEL", "decima:guard-small")
        monkeypatch.setenv("GUARD_TIMEOUT", "1.5")
        monkeypatch.setenv("GUARD_TOPIC_THRESHOLD", "0.7")
        monkeypatch.setenv("GUARD_JAILBREAK_THRESHOLD", "0.6")
        monkeypatch.setenv("GUARD_MAX_CHARS", "900")
        monkeypatch.setenv("GUARD_REFUSAL_MESSAGE", "Nope.")

        config = Config(env_path=tmp_path / "missing.env")

        assert config.guard_enabled is True
        assert config.guard_backend_url == "http://guard:9000"
        assert config.guard_api_key == "secret"
        assert config.guard_model == "decima:guard-small"
        assert config.guard_timeout == 1.5
        assert config.guard_topic_threshold == 0.7
        assert config.guard_jailbreak_threshold == 0.6
        assert config.guard_max_chars == 900
        assert config.guard_refusal_message == "Nope."

    def test_openreview_settings_from_env(self, tmp_path, monkeypatch):
        """Test OpenReview credentials load from the environment."""
        for key in ("OPENREVIEW_EMAIL", "OPENREVIEW_PASSWORD", "OPENREVIEW_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENREVIEW_EMAIL", "user@example.org")
        monkeypatch.setenv("OPENREVIEW_PASSWORD", "secret")
        monkeypatch.setenv("OPENREVIEW_TOKEN", "token-123")

        config = Config(env_path=tmp_path / "missing.env")

        assert config.openreview_email == "user@example.org"
        assert config.openreview_password == "secret"
        assert config.openreview_token == "token-123"

    def test_config_to_dict_hides_openreview_credentials(self, tmp_path):
        """Test that to_dict masks OpenReview password and token."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENREVIEW_EMAIL=user@example.org\n" "OPENREVIEW_PASSWORD=secret\n" "OPENREVIEW_TOKEN=token-123\n"
        )

        config = Config(env_path=env_file)
        config_dict = config.to_dict()

        assert config_dict["openreview_email"] == "user@example.org"
        assert config_dict["openreview_password"] == "***"
        assert config_dict["openreview_token"] == "***"
        assert config.openreview_password == "secret"
        assert config.openreview_token == "token-123"

    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "data_dir" in config_dict
        assert "chat_model" in config_dict
        assert "embedding_model" in config_dict
        assert "llm_backend_url" in config_dict
        assert "embedding_db" in config_dict
        assert "database_url" in config_dict
        assert "collection_name" in config_dict
        assert "max_context_papers" in config_dict
        assert "metadata_cache_ttl" in config_dict
        assert "chat_history_ttl" in config_dict
        assert "llm_connect_timeout" in config_dict
        assert "llm_request_timeout" in config_dict
        assert "web_ui_threads" in config_dict
        assert "guard_enabled" in config_dict
        assert "guard_backend_url" in config_dict
        assert "guard_model" in config_dict
        assert "guard_api_key" in config_dict

    def test_config_to_dict_hides_token(self, tmp_path):
        """Test that to_dict hides auth token."""
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_BACKEND_AUTH_TOKEN=secret-token-123")

        config = Config(env_path=env_file)
        config_dict = config.to_dict()

        # Token should be masked
        assert config_dict["llm_backend_auth_token"] == "***"
        # But original value is still accessible
        assert config.llm_backend_auth_token == "secret-token-123"

    def test_config_to_dict_hides_guard_api_key(self, tmp_path):
        """Test that to_dict hides the guard API key."""
        env_file = tmp_path / ".env"
        env_file.write_text("GUARD_API_KEY=guard-secret")

        config = Config(env_path=env_file)
        config_dict = config.to_dict()

        assert config_dict["guard_api_key"] == "***"
        assert config.guard_api_key == "guard-secret"

    def test_config_repr(self):
        """Test configuration string representation."""
        config = Config()
        repr_str = repr(config)

        assert "Config(" in repr_str
        assert "chat_model=" in repr_str
        assert "embedding_model=" in repr_str

    def test_config_data_dir(self, tmp_path):
        """Test that data_dir can be configured."""
        env_file = tmp_path / ".env"
        env_file.write_text("DATA_DIR=/custom/data")

        config = Config(env_path=env_file)

        assert config.data_dir == "/custom/data"
        # Paths should be resolved relative to custom data_dir
        assert config.embedding_db == str((Path("/custom/data") / "chroma_db").absolute())
        # database_url should be constructed from PAPER_DB (defaults to abstracts.db in custom data_dir)
        assert config.database_url == f"sqlite:///{(Path('/custom/data') / 'abstracts.db').absolute()}"

    def test_config_absolute_paths_unchanged(self, tmp_path):
        """Test that absolute paths are not modified."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
EMBEDDING_DB=/absolute/path/to/chroma_db
PAPER_DB=/absolute/path/to/papers.db
""")

        config = Config(env_path=env_file)

        # Absolute paths should remain unchanged
        assert config.embedding_db == str((Path("/absolute/path/to/chroma_db")).absolute())
        assert config.database_url == f"sqlite:///{Path('/absolute/path/to/papers.db').absolute()}"

    def test_config_embedding_db_url(self, tmp_path):
        """Test that EMBEDDING_DB can be a URL."""
        env_file = tmp_path / ".env"
        env_file.write_text("EMBEDDING_DB=http://chromadb:8000")

        config = Config(env_path=env_file)

        # URLs should remain unchanged
        assert config.embedding_db == "http://chromadb:8000"

    def test_config_postgres_components_assemble_url(self, tmp_path):
        """POSTGRES_* + ABSTRACTS_DB_PASSWORD assembles a PostgreSQL URL when PAPER_DB is unset."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "POSTGRES_USER=abstracts\n"
            "POSTGRES_HOST=abstracts-postgres\n"
            "POSTGRES_PORT=5432\n"
            "POSTGRES_DB=abstracts\n"
            "ABSTRACTS_DB_PASSWORD=secret123\n"
        )

        config = Config(env_path=env_file)

        assert config.database_url == "postgresql://abstracts:secret123@abstracts-postgres:5432/abstracts"

    def test_config_postgres_components_ignored_when_paper_db_is_url(self, tmp_path):
        """POSTGRES_* vars are ignored when PAPER_DB already contains a full URL."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "PAPER_DB=postgresql://user:pass@host:5432/db\n"
            "POSTGRES_USER=abstracts\n"
            "POSTGRES_HOST=abstracts-postgres\n"
            "POSTGRES_PORT=5432\n"
            "POSTGRES_DB=abstracts\n"
            "ABSTRACTS_DB_PASSWORD=secret123\n"
        )

        config = Config(env_path=env_file)

        # Explicit PAPER_DB takes precedence
        assert config.database_url == "postgresql://user:pass@host:5432/db"

    def test_config_postgres_components_partial_does_not_assemble(self, tmp_path):
        """Partial POSTGRES_* vars (missing password) fall back to SQLite."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "POSTGRES_USER=abstracts\n"
            "POSTGRES_HOST=abstracts-postgres\n"
            # ABSTRACTS_DB_PASSWORD deliberately omitted
        )

        config = Config(env_path=env_file)

        # Falls back to the default SQLite path
        assert config.database_url.startswith("sqlite:///")

    def test_config_postgres_env_overrides_paper_db_from_file(self, tmp_path, monkeypatch):
        """POSTGRES_HOST in os.environ overrides PAPER_DB URL baked into a .env file.

        This simulates the Podman quadlet scenario where the container image
        ships a default /app/.env with PAPER_DB pointing to the Docker Compose
        service name ('postgres'), but the quadlet EnvironmentFile injects
        POSTGRES_HOST=abstracts-postgres (and friends) at runtime.  The runtime
        env vars must win so the application connects to the correct host.
        """
        # Simulate the baked-in /app/.env that ships inside the container image.
        env_file = tmp_path / ".env"
        env_file.write_text("PAPER_DB=postgresql://abstracts:old_password@postgres:5432/abstracts\n")

        # Simulate the quadlet EnvironmentFile injecting POSTGRES_* vars at runtime.
        monkeypatch.setenv("POSTGRES_HOST", "abstracts-postgres")
        monkeypatch.setenv("POSTGRES_USER", "abstracts")
        monkeypatch.setenv("ABSTRACTS_DB_PASSWORD", "secret123")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DB", "abstracts")

        config = Config(env_path=env_file)

        # Runtime POSTGRES_* vars must override the PAPER_DB baked into the image.
        assert config.database_url == "postgresql://abstracts:secret123@abstracts-postgres:5432/abstracts"

    def test_config_default_conference_and_year(self, tmp_path):
        """Test that DEFAULT_CONFERENCE and DEFAULT_YEAR can be configured."""
        env_file = tmp_path / ".env"
        env_file.write_text("""
DEFAULT_CONFERENCE=NeurIPS
DEFAULT_YEAR=2024
""")

        config = Config(env_path=env_file)

        assert config.default_conference == "NeurIPS"
        assert config.default_year == 2024

    def test_config_default_conference_and_year_defaults(self, tmp_path):
        """Test that DEFAULT_CONFERENCE defaults to 'ML4PS' and DEFAULT_YEAR to 0."""
        env_file = tmp_path / ".env"
        env_file.write_text("")

        config = Config(env_path=env_file)

        assert config.default_conference == "ML4PS"
        assert config.default_year == 0

    def test_config_imprint_link_default(self, tmp_path):
        """Test that imprint_link defaults to empty string."""
        env_file = tmp_path / ".env"
        env_file.write_text("")

        config = Config(env_path=env_file)

        assert config.imprint_link == ""

    def test_config_imprint_link_from_env(self, tmp_path):
        """Test loading IMPRINT_LINK from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("IMPRINT_LINK=https://example.com/imprint")

        config = Config(env_path=env_file)

        assert config.imprint_link == "https://example.com/imprint"


class TestGetConfig:
    """Test get_config function."""

    def teardown_method(self):
        """Reset global config state after each test."""
        config_module._config = None

    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_reload(self, tmp_path, monkeypatch):
        """Test that reload=True creates new instance."""
        # Get initial config
        config1 = get_config()

        # Change environment variable
        monkeypatch.setenv("CHAT_MODEL", "reloaded-model")

        # Get config without reload - should be same instance
        config2 = get_config(reload=False)
        assert config2 is config1
        assert config2.chat_model != "reloaded-model"

        # Get config with reload - should be new instance
        config3 = get_config(reload=True)
        assert config3.chat_model == "reloaded-model"

    def test_get_config_returns_config_instance(self):
        """Test that get_config returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)
