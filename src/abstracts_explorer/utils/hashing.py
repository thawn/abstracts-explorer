"""
Content hashing helpers.

These helpers produce the SHA-256 content hashes stored in ChromaDB and in the
paper database. They live outside the embeddings package to avoid import cycles.
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


def paper_content_hash(title: Optional[str], abstract: Optional[str]) -> str:
    """
    Return the content hash for a paper title and abstract.

    This mirrors the text used for embeddings (``title + "\\n\\n" + abstract``)
    so the hash stored in the database matches the hash stored in ChromaDB.

    Parameters
    ----------
    title : str or None
        Paper title.
    abstract : str or None
        Paper abstract.

    Returns
    -------
    str
        ``sha256:`` followed by the hexadecimal SHA-256 digest.
    """
    text = f"{title or ''}\n\n{abstract or ''}".strip()
    return hash_text(text)


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
