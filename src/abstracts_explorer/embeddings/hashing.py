"""
Content hashing helpers.

ChromaDB stores only paper UIDs, vectors, and the SHA-256 hash of the embedded
text; these helpers produce and recognise those hashes.
"""

import hashlib
import re
from typing import Optional

HASH_PREFIX = "sha256:"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def hash_text(text: str) -> str:
    """
    Return the content hash used to identify embedded text.

    Parameters
    ----------
    text : str
        Text that was embedded.

    Returns
    -------
    str
        ``sha256:`` followed by the hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def is_content_hash(value: Optional[str]) -> bool:
    """
    Return whether *value* is a content hash produced by :func:`hash_text`.

    Parameters
    ----------
    value : str or None
        Candidate ChromaDB document value.

    Returns
    -------
    bool
        ``True`` when *value* matches the ``sha256:<hex>`` format.
    """
    return bool(value) and bool(_HASH_RE.match(str(value)))
