"""
Batch-size constants and iteration helpers.
"""

from collections.abc import Iterator, Sequence
from typing import Any

from abstracts_explorer.constants import MAX_BIND_PARAMETERS

# Maximum number of results to request from ChromaDB in a single query.
# ChromaDB bundles SQLite >= 3.32, which accepts up to 32766 bound parameters.
MAX_QUERY_RESULTS = MAX_BIND_PARAMETERS

IO_BATCH_SIZE = 1000

# Maximum number of full embedding vectors fetched from ChromaDB in one call.
# Bounds the transient payload of collection.get(include=["embeddings"]).
EMBEDDING_FETCH_BATCH_SIZE = 5000

EMBED_BATCH_SIZE = 128


def iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive slices of *items* with at most *size* elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
