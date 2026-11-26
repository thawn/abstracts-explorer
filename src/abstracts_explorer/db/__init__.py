"""
Database package.

``DatabaseManager`` is composed from per-responsibility mixins (papers,
search, catalog, caches, donations, evaluation, transfer) and re-exported
here together with the errors and module-level helpers.  The public import
surface remains ``abstracts_explorer.database``.
"""

from abstracts_explorer.db_models import Base
from abstracts_explorer.db.errors import (
    DatabaseError,
    EmbeddingModelConflictError,
    normalize_model_name,
)
from abstracts_explorer.db.manager import DatabaseManager
from abstracts_explorer.db.session import (
    mask_database_url,
    reset_metadata_cache,
    reset_schema_cache,
)
from abstracts_explorer.utils.read_only import ReadOnlyError

__all__ = [
    "Base",
    "DatabaseError",
    "DatabaseManager",
    "EmbeddingModelConflictError",
    "ReadOnlyError",
    "normalize_model_name",
    "mask_database_url",
    "reset_metadata_cache",
    "reset_schema_cache",
]
