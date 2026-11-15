"""
Embeddings package.

Generates text embeddings for paper abstracts and stores them in ChromaDB.
ChromaDB stores only paper UIDs, vectors, and a SHA-256 hash of the embedded
text; all paper metadata lives in the SQL database.

Public API is re-exported here so ``from abstracts_explorer.embeddings import
...`` keeps working.
"""

from abstracts_explorer.embeddings.errors import EmbeddingsError
from abstracts_explorer.embeddings.hashing import HASH_PREFIX, hash_text, is_content_hash
from abstracts_explorer.embeddings.manager import EmbeddingsManager

__all__ = [
    "EmbeddingsManager",
    "EmbeddingsError",
    "HASH_PREFIX",
    "hash_text",
    "is_content_hash",
]
