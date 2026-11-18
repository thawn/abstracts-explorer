"""
Batch-size constants and iteration helpers.
"""

from collections.abc import Iterator, Sequence
from typing import Any

# Maximum number of results to request from ChromaDB in a single query.
# Prevents "too many SQL variables" errors in the underlying SQLite backend.
MAX_QUERY_RESULTS = 32766  # this is the maximum for sqlite 3.32 and above

IO_BATCH_SIZE = 1000
ID_BATCH_SIZE = 500
EMBED_BATCH_SIZE = 32


def iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive slices of *items* with at most *size* elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
