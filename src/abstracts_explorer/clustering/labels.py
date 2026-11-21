"""
Helpers for cluster legend labels and descriptive sentences.
"""

import json
import re
from typing import Any, Tuple

LABEL_SEPARATOR = "\u241f"
MAX_LABEL_CHARS = 120
MAX_DESCRIPTION_CHARS = 240

_LABEL_KEYS = ("label", "topic_label", "short_label")
_DESCRIPTION_KEYS = ("description", "summary", "sentence")

_LABEL_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}\s*(?:label|short label|topic label)\s*\*{0,2}\s*:\s*\*{0,2}\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DESCRIPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}\s*(?:description|summary|sentence)\s*\*{0,2}\s*:\s*\*{0,2}\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_text(value: Any) -> str:
    """Collapse whitespace and strip surrounding quotes or emphasis marks."""
    return " ".join(str(value).split()).strip("\"'*").strip()


def clean_label(value: Any) -> str:
    """
    Normalize and truncate a short cluster label.

    Parameters
    ----------
    value : Any
        Raw label value.

    Returns
    -------
    str
        Whitespace-normalized label without a ``topic:`` prefix, capped at
        :data:`MAX_LABEL_CHARS`.
    """
    text = _normalize_text(value)
    if text.lower().startswith("topic:"):
        text = text.split(":", 1)[1].strip()
    return text[:MAX_LABEL_CHARS]


def clean_description(value: Any) -> str:
    """
    Normalize and truncate a cluster description.

    Parameters
    ----------
    value : Any
        Raw description value.

    Returns
    -------
    str
        Whitespace-normalized description capped at
        :data:`MAX_DESCRIPTION_CHARS`.
    """
    return _normalize_text(value)[:MAX_DESCRIPTION_CHARS]


def parse_llm_label_response(response: str) -> Tuple[str, str]:
    """
    Parse a two-line LLM label response.

    Supports ``Label:`` and ``Description:``/``Summary:`` lines as well as
    JSON objects. Plain single-line responses are treated as a legacy label.

    Parameters
    ----------
    response : str
        Raw model response.

    Returns
    -------
    tuple of str
        ``(label, description)``. Either value may be empty when absent.
    """
    text = (response or "").strip()
    if not text:
        return "", ""

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            raw_label = next((payload[key] for key in _LABEL_KEYS if payload.get(key)), "")
            raw_description = next((payload[key] for key in _DESCRIPTION_KEYS if payload.get(key)), "")
            label = clean_label(raw_label) if raw_label else ""
            description = clean_description(raw_description) if raw_description else ""
            return label, description

    label_match = _LABEL_LINE_RE.search(text)
    description_match = _DESCRIPTION_LINE_RE.search(text)
    if label_match or description_match:
        label = clean_label(label_match.group(1)) if label_match else ""
        description = clean_description(description_match.group(1)) if description_match else ""
        return label, description

    return clean_label(text), ""


def combine_label_description(label: Any, description: Any) -> str:
    """
    Encode a label and description into BERTopic's single-string format.

    Parameters
    ----------
    label : Any
        Short label.
    description : Any
        Descriptive sentence.

    Returns
    -------
    str
        ``label`` alone when no description exists, otherwise both values
        joined with :data:`LABEL_SEPARATOR`.
    """
    clean = clean_label(label)
    sentence = clean_description(description)
    if sentence:
        return f"{clean}{LABEL_SEPARATOR}{sentence}"
    return clean


def split_label_description(value: Any) -> Tuple[str, str]:
    """
    Split an encoded label/description pair.

    Plain legacy labels return an empty description.

    Parameters
    ----------
    value : Any
        Encoded or plain label value.

    Returns
    -------
    tuple of str
        ``(label, description)``.
    """
    text = _normalize_text(value)
    if LABEL_SEPARATOR in text:
        label, description = text.split(LABEL_SEPARATOR, 1)
        return clean_label(label), clean_description(description)
    return clean_label(text), ""
