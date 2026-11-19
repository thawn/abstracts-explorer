"""
Embedding errors and error translation helpers.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EmbeddingsError(Exception):
    """Exception raised for embedding operations."""


@contextmanager
def translate_errors(message: str) -> Generator[None, None, None]:
    """Wrap unexpected exceptions in :class:`EmbeddingsError`, passing existing ones through."""
    try:
        yield
    except EmbeddingsError:
        raise
    except Exception as e:
        raise EmbeddingsError(f"{message}: {str(e)}") from e
