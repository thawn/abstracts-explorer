"""
Database exceptions and helpers.
"""

import re


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class EmbeddingModelConflictError(DatabaseError):
    """
    Raised when the embedding model in imported data differs from the local database.

    Attributes
    ----------
    local_model : str
        Embedding model currently in the local database.
    remote_model : str
        Embedding model in the data being imported.
    """

    def __init__(self, local_model: str, remote_model: str) -> None:
        self.local_model = local_model
        self.remote_model = remote_model
        super().__init__(
            f"Embedding model mismatch: local database uses '{local_model}' "
            f"but imported data uses '{remote_model}'. Cannot import data "
            f"created with a different embedding model."
        )


def normalize_model_name(name: str) -> str:
    """
    Normalize an embedding model name for comparison.

    Strips a leading ``alias-`` prefix (case-insensitive) so that, e.g.,
    ``alias-qwen3-embeddings-8b`` is considered identical to
    ``qwen3-embeddings-8b``.

    Parameters
    ----------
    name : str
        Embedding model name.

    Returns
    -------
    str
        Normalized model name.
    """
    return re.sub(r"^alias-", "", name, flags=re.IGNORECASE)
