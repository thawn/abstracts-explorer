"""
Tests for configuration module.

This module tests the configuration loading and management functionality.
"""

import os
from pathlib import Path

from abstracts_explorer.config import Config, get_config, load_env_file
from abstracts_explorer import config as config_module

from tests.conftest import get_env_test_path


class TestLoadEnvFile:
    """Test .env file loading."""

    def test_load_env_file_basic(self, tmp_path):
        """Test loading a basic .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
# Comment line
CHAT_MODEL=test-model
EMBEDDING_MODEL=test-embedding
LLM_BACKEND_URL=http://test:8080

# Another comment
MAX_CONTEXT_PAPERS=10
"""
        )

        env_vars = load_env_file(env_file)

        assert env_vars["CHAT_MODEL"] == "test-model"
        assert env_vars["EMBEDDING_MODEL"] == "test-embedding"
        assert env_vars["LLM_BACKEND_URL"] == "http://test:8080"
        assert env_vars["MAX_CONTEXT_PAPERS"] == "10"

    def test_load_env_file_with_quotes(self, tmp_path):
        """Test loading .env file with quoted values."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
CHAT_MODEL="quoted-model"
EMBEDDING_MODEL='single-quoted'
LLM_BACKEND_URL=http://localhost:1234
"""
        )

        env_vars = load_env_file(env_file)

        assert env_vars["CHAT_MODEL"] == "quoted-model"
        assert env_vars["EMBEDDING_MODEL"] == "single-quoted"
        assert env_vars["LLM_BACKEND_URL"] == "http://localhost:1234"

    def test_load_env_file_empty_lines(self, tmp_path):
        """Test loading .env file with empty lines."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
CHAT_MODEL=model1

EMBEDDING_MODEL=model2


LLM_BACKEND_URL=http://test
"""
        )

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


class TestConfig:
    """Test Config class."""

    def setup_method(self):
        """Clear environment variables before each test to ensure isolation."""
        # List of all config-related environment variables
        env_vars_to_clean = [
            "DATA_DIR",
            "CHAT_MODEL",
            "EMBEDDING_MODEL",
            "LLM_BACKEND_URL",
            "LLM_BACKEND_AUTH_TOKEN",
            "EMBEDDING_DB",
            "PAPER_DB",
            "COLLECTION_NAME",
            "MAX_CONTEXT_PAPERS",
            "CHAT_TEMPERATURE",
            "CHAT_MAX_TOKENS",
            "ENABLE_QUERY_REWRITING",
            "QUERY_SIMILARITY_THRESHOLD",
        ]
        for var in env_vars_to_clean:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Clean up environment variables after each test."""
        # List of all config-related environment variables
        env_vars_to_clean = [
            "DATA_DIR",
            "CHAT_MODEL",
            "EMBEDDING_MODEL",
            "LLM_BACKEND_URL",
            "LLM_BACKEND_AUTH_TOKEN",
            "EMBEDDING_DB",
            "PAPER_DB",
            "COLLECTION_NAME",
            "MAX_CONTEXT_PAPERS",
            "CHAT_TEMPERATURE",
            "CHAT_MAX_TOKENS",
            "ENABLE_QUERY_REWRITING",
            "QUERY_SIMILARITY_THRESHOLD",
        ]
        for var in env_vars_to_clean:
            if var in os.environ:
                del os.environ[var]

    def test_config_defaults(self, monkeypatch):
        """Test default configuration values."""
        # Clear all environment variables that might override defaults
        env_vars_to_clear = [
            "CHAT_MODEL",
            "EMBEDDING_MODEL",
            "LLM_BACKEND_URL",
            "LLM_BACKEND_AUTH_TOKEN",
            "EMBEDDING_DB",
            "PAPER_DB",
            "COLLECTION_NAME",
            "MAX_CONTEXT_PAPERS",
            "CHAT_TEMPERATURE",
            "CHAT_MAX_TOKENS",
            "DATA_DIR",
        ]
        for var in env_vars_to_clear:
            monkeypatch.delenv(var, raising=False)

        # Use .env.test which has the default values
        config = Config(env_path=get_env_test_path())

        assert config.data_dir == "data"
        assert config.chat_model == "alias-fast"
        assert config.embedding_model == "alias-qwen3-8b-embeddings"
        assert config.llm_backend_url == "https://api.helmholtz-blablador.fz-juelich.de"
        assert config.embedding_db == str((Path("data") / "chroma_db").absolute())
        # Test database_url is constructed from PAPER_DB
        assert config.database_url == f"sqlite:///{(Path('data') / 'abstracts.db').absolute()}"
        assert config.collection_name == "papers"
        assert config.max_context_papers == 5
        assert config.chat_temperature == 0.7
        assert config.chat_max_tokens == 2000

    def test_config_from_env_file(self, tmp_path):
        """Test loading configuration from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
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
"""
        )

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
        env_file.write_text(
            """
MAX_CONTEXT_PAPERS=20
CHAT_TEMPERATURE=0.5
CHAT_MAX_TOKENS=500
"""
        )

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
        env_file.write_text(
            """
EMBEDDING_DB=/absolute/path/to/chroma_db
PAPER_DB=/absolute/path/to/papers.db
"""
        )

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
