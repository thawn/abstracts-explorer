"""
LLM Utilities
=============

Shared helpers for one-shot OpenAI-compatible chat completion calls.

Both the clustering label generation and the evaluation pipeline talk to the
same OpenAI-compatible back-end with the same ``system``/``user`` message
shape.  This module centralizes that call so request construction stays
consistent across modules.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def chat_completion_text(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Make a one-shot chat completion call and return the response text.

    Parameters
    ----------
    client
        OpenAI-compatible client exposing ``chat.completions.create``.
    model : str
        Model identifier to use for the call.
    system_prompt : str
        System message content.
    user_prompt : str
        User message content.
    temperature : float, optional
        Sampling temperature, by default 0.3.
    max_tokens : int, optional
        Maximum tokens to generate, by default 1000.
    timeout : float, optional
        Request timeout in seconds; omitted from the request when None.
    reasoning_effort : str, optional
        Reasoning-effort hint for models that support it (e.g. ``"low"``);
        omitted from the request when None.

    Returns
    -------
    str
        The model's response text, stripped of surrounding whitespace.
        Empty when the model returned no content.
    """
    extra_kwargs: dict = {}
    if timeout is not None:
        extra_kwargs["timeout"] = timeout
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_kwargs,
    )

    return (response.choices[0].message.content or "").strip()
