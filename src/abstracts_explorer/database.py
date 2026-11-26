"""
Database facade.

The implementation lives in the :mod:`abstracts_explorer.db` package; this
module re-exports its public API so existing imports
(``from abstracts_explorer.database import DatabaseManager``) keep working.
"""

from abstracts_explorer.db import (
    Base,
    DatabaseError,
    DatabaseManager,
    EmbeddingModelConflictError,
    ReadOnlyError,
    normalize_model_name,
    reset_metadata_cache,
    reset_schema_cache,
)

__all__ = [
    "Base",
    "DatabaseError",
    "DatabaseManager",
    "EmbeddingModelConflictError",
    "ReadOnlyError",
    "normalize_model_name",
    "reset_metadata_cache",
    "reset_schema_cache",
]
