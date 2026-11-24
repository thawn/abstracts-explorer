"""
LLM helpers for cluster labeling.

HTTP retries (including ``429``/``Retry-After``) are owned by the OpenAI SDK
through the rate-limit-aware transport. This module only retries responses that
contain no usable content, doubling the token budget each time.
"""

import logging
import time
from typing import Any, Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.llm_utils import chat_completion_text

logger = logging.getLogger(__name__)

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


def generate_llm_label(
    embeddings_manager: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_LABEL_MAX_TOKENS,
) -> str:
    """
    Generate one cluster label, retrying empty or unusable responses.

    The OpenAI SDK retries HTTP failures (status codes, ``Retry-After``,
    backoff, attempt budget) before this function sees them. Here, a response
    without usable content is retried with a doubled token budget, up to
    :data:`MAX_LABEL_MAX_TOKENS`.

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
        If the HTTP client raised a non-empty-response error, or all attempts
        returned no usable content.
    """
    if not hasattr(embeddings_manager, "openai_client"):
        raise AttributeError("OpenAI client not available in embeddings manager")

    config = get_config()
    max_attempts = max(1, config.llm_max_attempts)
    retry_base_delay = config.llm_retry_base_delay
    retry_max_delay = config.llm_retry_max_delay

    empty_failures = 0
    for attempt in range(1, max_attempts + 1):
        current_max_tokens = min(max_tokens * (2**empty_failures), MAX_LABEL_MAX_TOKENS)
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
            error: Optional[Exception] = None
        except ValueError as exc:
            content = ""
            error = exc
        except Exception as exc:
            status_code = _status_code(exc)
            logger.error(
                "Cluster-label LLM call failed permanently (model=%s, attempt %d/%d, status=%s): %s",
                config.cluster_label_model,
                attempt,
                max_attempts,
                status_code if status_code is not None else "n/a",
                exc,
            )
            raise

        if content.strip():
            return content.strip().strip("\"'")

        error = error or ValueError("LLM returned empty content")
        empty_failures += 1
        if attempt >= max_attempts:
            logger.error(
                "Cluster-label LLM returned no usable content permanently "
                "(model=%s, attempt %d/%d, max_tokens=%d): %s",
                config.cluster_label_model,
                attempt,
                max_attempts,
                current_max_tokens,
                error,
            )
            raise error

        delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
        logger.warning(
            "Cluster-label LLM returned no usable content (model=%s, attempt %d/%d, "
            "max_tokens=%d): %s; retrying in %.1fs",
            config.cluster_label_model,
            attempt,
            max_attempts,
            current_max_tokens,
            error,
            delay,
        )
        time.sleep(delay)

    raise ValueError("LLM returned empty content")
