"""
Token budgeting and truncation helpers for embedding inputs.
"""

import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TOKEN_SAFETY_MARGIN_RATIO = 0.1
_CHAR_FALLBACK_RATIO = 2
_MAX_CONTEXT_RETRIES = 3
_CONTEXT_RETRY_SHRINK = 0.75


def token_margin(limit: int) -> int:
    """Return the safety margin applied to a token limit."""
    return max(1, int(limit * _TOKEN_SAFETY_MARGIN_RATIO))


@lru_cache(maxsize=1)
def _get_token_encoding() -> Optional[Any]:
    """Return a cached tiktoken encoding, or ``None`` when unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.debug("tiktoken unavailable (%s); falling back to character truncation", exc)
        return None


def count_tokens(text: str) -> Optional[int]:
    """Count tokens with the cached encoding, returning ``None`` without one."""
    encoding = _get_token_encoding()
    if encoding is None:
        return None
    return len(encoding.encode(text))


def truncate_with_counts(text: str, max_tokens: int) -> tuple[str, Optional[int], Optional[int]]:
    """
    Truncate *text* to at most *max_tokens* tokens in a single encode/decode pass.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_tokens : int
        Maximum number of tokens. Non-positive values disable truncation.

    Returns
    -------
    tuple of (str, int or None, int or None)
        The (possibly truncated) text, the original token count, and the
        truncated token count. Token counts are ``None`` when tiktoken is
        unavailable or truncation is disabled.
    """
    if max_tokens <= 0 or not text:
        return text, None, None

    encoding = _get_token_encoding()
    if encoding is None:
        return text[: max_tokens * _CHAR_FALLBACK_RATIO], None, None

    tokens = encoding.encode(text)
    original_count = len(tokens)
    if original_count <= max_tokens:
        return text, original_count, original_count
    return encoding.decode(tokens[:max_tokens]), original_count, max_tokens


def is_context_length_error(error: Exception) -> bool:
    """Return ``True`` when *error* reports an exceeded input context window."""
    message = str(error).lower()
    markers = (
        "maximum context length",
        "context window",
        "context length",
        "too many tokens",
        "reduce the length of the input",
    )
    return any(marker in message for marker in markers)


def context_length_limit_from_error(error: Exception) -> Optional[int]:
    """Extract the model's maximum context length from an API error, if present."""
    import re

    match = re.search(r"maximum context length is (\d+)", str(error), re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))
