"""
Retry-aware LLM helpers for cluster labeling.
"""

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.llm_utils import chat_completion_text
from abstracts_explorer.rate_limiter import RateLimitCooldown

logger = logging.getLogger(__name__)

_RETRYABLE_LLM_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_rate_limit_cooldown = RateLimitCooldown()

DEFAULT_LABEL_MAX_TOKENS = 600
MAX_LABEL_MAX_TOKENS = 2400


def _status_code(error: Exception) -> Optional[int]:
    """Return an HTTP status code from an LLM client error, if present."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(error: Exception) -> Optional[float]:
    """Parse a ``Retry-After`` header from an LLM client error."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def is_retryable_llm_error(error: Exception) -> bool:
    """
    Return whether a cluster-label LLM error should be retried.

    Parameters
    ----------
    error : Exception
        Exception raised by the LLM client.

    Returns
    -------
    bool
        ``True`` for empty responses, transient network errors, and retryable
        HTTP statuses.
    """
    if isinstance(error, AttributeError):
        return False
    if isinstance(error, ValueError):
        return True

    status_code = _status_code(error)
    if status_code is None:
        return True
    return status_code in _RETRYABLE_LLM_STATUS_CODES


def generate_llm_label(
    embeddings_manager: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_LABEL_MAX_TOKENS,
) -> str:
    """
    Generate one cluster label with retries.

    Empty responses, transient network errors, and retryable HTTP statuses are
    retried with exponential backoff. Non-retryable failures and exhausted
    retries are raised to the caller. After each empty response the token
    budget doubles, up to :data:`MAX_LABEL_MAX_TOKENS`.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager exposing an ``openai_client``.
    system_prompt : str
        System message for the LLM.
    user_prompt : str
        User message for the LLM.
    max_tokens : int, optional
        Base maximum number of tokens to generate (default
        :data:`DEFAULT_LABEL_MAX_TOKENS`). Doubles after every empty response,
        capped at :data:`MAX_LABEL_MAX_TOKENS`.

    Returns
    -------
    str
        The model's response, stripped of whitespace and surrounding quotes.

    Raises
    ------
    AttributeError
        If the OpenAI client is not available.
    Exception
        If all retry attempts fail.
    """
    if not hasattr(embeddings_manager, "openai_client"):
        raise AttributeError("OpenAI client not available in embeddings manager")

    config = get_config()
    max_attempts = max(1, config.llm_max_attempts)
    retry_base_delay = config.llm_retry_base_delay
    retry_max_delay = config.llm_retry_max_delay

    last_error: Optional[Exception] = None
    empty_failures = 0
    for attempt in range(1, max_attempts + 1):
        current_max_tokens = min(max_tokens * (2**empty_failures), MAX_LABEL_MAX_TOKENS)
        _rate_limit_cooldown.wait()
        try:
            content = chat_completion_text(
                client=embeddings_manager.openai_client,
                model=config.cluster_label_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=current_max_tokens,
                timeout=config.llm_request_timeout,
                reasoning_effort="low",
            )
            if not content:
                empty_failures += 1
                raise ValueError("LLM returned empty content")
            return content.strip().strip("\"'")
        except Exception as error:
            last_error = error
            if attempt >= max_attempts or not is_retryable_llm_error(error):
                raise

            if _status_code(error) == 429:
                retry_after = _retry_after_seconds(error)
                delay = retry_after if retry_after is not None else config.llm_rate_limit_delay
                _rate_limit_cooldown.defer(delay)
                logger.warning(
                    "Cluster-label LLM call rate-limited (attempt %d/%d, max_tokens=%d): %s; "
                    "pausing all LLM label calls for %.1fs",
                    attempt,
                    max_attempts,
                    current_max_tokens,
                    error,
                    delay,
                )
            else:
                delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
                logger.warning(
                    "Cluster-label LLM call failed (attempt %d/%d, max_tokens=%d): %s; " "retrying in %.1fs",
                    attempt,
                    max_attempts,
                    current_max_tokens,
                    error,
                    delay,
                )
            time.sleep(delay)

    assert last_error is not None
    raise last_error
