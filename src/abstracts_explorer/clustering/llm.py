"""
Retry-aware LLM helpers for cluster labeling.
"""

import logging
import time
from typing import Any, Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.llm_utils import chat_completion_text

logger = logging.getLogger(__name__)

_RETRYABLE_LLM_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


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

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code is None:
        return True

    try:
        return int(status_code) in _RETRYABLE_LLM_STATUS_CODES
    except (TypeError, ValueError):
        return True


def generate_llm_label(
    embeddings_manager: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 200,
) -> str:
    """
    Generate one cluster label with retries.

    Empty responses, transient network errors, and retryable HTTP statuses are
    retried with exponential backoff. Non-retryable failures and exhausted
    retries are raised to the caller.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager exposing an ``openai_client``.
    system_prompt : str
        System message for the LLM.
    user_prompt : str
        User message for the LLM.
    max_tokens : int, optional
        Maximum number of tokens to generate (default 200).

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
    for attempt in range(1, max_attempts + 1):
        try:
            content = chat_completion_text(
                client=embeddings_manager.openai_client,
                model=config.cluster_label_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=config.llm_request_timeout,
                reasoning_effort="low",
            )
            if not content:
                raise ValueError("LLM returned empty content")
            return content.strip().strip("\"'")
        except Exception as error:
            last_error = error
            if attempt >= max_attempts or not is_retryable_llm_error(error):
                raise
            delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
            logger.warning(
                "Cluster-label LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                max_attempts,
                error,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error
