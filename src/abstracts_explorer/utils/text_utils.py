"""
Text-normalization helpers shared across the package.
"""

from typing import Any, Optional

from bs4 import BeautifulSoup


def strip_nul_bytes(value: Any) -> str:
    """
    Remove NUL (0x00) bytes from a value.

    PostgreSQL ``text``/``varchar`` columns reject NUL bytes, so text coming
    from external sources (downloads, registry artifacts) must be cleaned
    before it is written to a PostgreSQL-backed database.

    Parameters
    ----------
    value : Any
        Text-like value.  ``None`` is treated as an empty string.

    Returns
    -------
    str
        The value as a string with every NUL byte removed.

    Examples
    --------
    >>> strip_nul_bytes("alpha\\x00beta")
    'alphabeta'
    >>> strip_nul_bytes(None)
    ''
    """
    if value is None:
        return ""
    return str(value).replace("\x00", "")


def strip_nul_bytes_optional(value: Optional[str]) -> Optional[str]:
    """
    Remove NUL (0x00) bytes from an optional string.

    Parameters
    ----------
    value : str or None
        Text value to clean.

    Returns
    -------
    str or None
        ``None`` is preserved; otherwise the string with NUL bytes removed.
    """
    if value is None:
        return None
    return str(value).replace("\x00", "")


def sanitize_json_strings(value: Any) -> Any:
    """
    Recursively strip NUL bytes from every string in a JSON-like structure.

    Parameters
    ----------
    value : Any
        A JSON-serialisable value (dict, list, str, or scalar).

    Returns
    -------
    Any
        A new structure with NUL bytes removed from all strings.  Keys are
        sanitized as well, since they may be written to text columns.

    Examples
    --------
    >>> sanitize_json_strings({"label": "a\\x00b", "items": ["c\\x00d"]})
    {'label': 'ab', 'items': ['cd']}
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {sanitize_json_strings(key): sanitize_json_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_json_strings(item) for item in value)
    return value


def html_to_text(value: Any) -> str:
    """
    Strip HTML tags, unescape entities, and normalize whitespace.

    Parameters
    ----------
    value : Any
        Text or HTML fragment.  ``None`` is treated as an empty string.

    Returns
    -------
    str
        Plain text with tags removed, HTML entities decoded, and repeated
        whitespace collapsed.

    Examples
    --------
    >>> html_to_text("<b>Data Management</b> Makes &amp; ML Easier")
    'Data Management Makes & ML Easier'
    >>> html_to_text(None)
    ''
    """
    if value is None:
        return ""

    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())
